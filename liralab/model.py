import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# using cloned code from https://github.com/facebookresearch/detr
# adding the dir to PYTHONPATH didn't work for some reason, some classes would import but others would not hence this nasty hack

from detr.models.transformer import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer
from detr.models.transformer import Transformer as DETRTransformer
from detr.models.position_encoding import PositionEmbeddingSine, PositionEmbeddingLearned, NestedTensor
import torch
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
import copy

#from act.utils import sample_box_pose, sample_insertion_pose # robot functions
#from act.visualize_episodes import save_videos

from pdb import set_trace
import matplotlib.pyplot as plt
import math
from typing import Optional, List
import h5py
from torch.utils.data import DataLoader
from pathlib import Path

import torch.nn as nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
from torch import nn, Tensor

#from act.sim_env import make_sim_env, BOX_POSE
from aim import Run

args = {
    'num_epochs': 4500,
    'eval_interval_epochs': 500,
    'num_eval_rollouts': 100,
    'lr_backbone': 1e-5,
    'batch_size': 8,
    'trained_model_dir': 'experiments/AAA/', # '../experiments/aloha/sim_transfer_cube_scripted/second_workstream/',
    'task_name': 'sim_transfer_cube_scripted',
    'dataset_dir': 'data/liralab/AAA/', # '../data/aloha/sim_transfer_cube_scripted',
    'chunk_size': 100,  # chunk_size is --> num_queries <-- !!!!
    'd_model': 512,  # d_model
    'dim_feedforward': 3200,
    'lr': 1e-5,
    'kl_weight': 10,
    'state_dim': 6,
    'action_dim': 6,
    'num_episodes': 2,
    'episode_len': 600,
    'camera_names': ['top'],
    'num_encoder_layers': 4,
    'num_decoder_layers': 7,
    'backbone': 'resnet18',
    'nhead': 8,
    'weight_decay': 1e-4,
    'dropout': 0.1,
    'position_embedding': 'sine',
    'normalize_before': False,
    'fps': 10,
    'latent_dim': 32,
}

def get_norm_stats(dataset_dir : str):
    dataset_dir = Path(dataset_dir)
    hdf5_files = list(dataset_dir.glob("*.hdf5"))
    all_qpos_data = []
    all_action_data = []
    for episode_path in hdf5_files:
        # dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(episode_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats

class Backbone(nn.Module):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name, position_embedding='sine'):
        super().__init__()
        backbone = getattr(torchvision.models, name)(pretrained=True)  # authors of ACT trained with frozen batchnorm, but not clear this should offer an advantage
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.num_channels = 512 if name in ('resnet18', 'resnet34') else 2048

        if position_embedding == 'sine':
            self.position_embedding = PositionEmbeddingSine(self.num_channels // 2, normalize=True)
        elif position_embedding == 'learned':
            self.position_embedding = PositionEmbeddingLearned(self.num_channels // 2)
        else:
            raise ValueError(f"not supported {position_embedding}")

    def forward(self, x):
        x = self.backbone(x)
        pos_embs = self.position_embedding(NestedTensor(x, torch.ones_like(x[0, [0]], dtype=torch.int8)))
        return NestedTensor(x, pos_embs)


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld

class Transformer(DETRTransformer):
    def forward(self, src, pos_embed, query_embed):
        memory = self.encoder(src, src_key_padding_mask=None, pos=pos_embed)

        tgt = torch.zeros_like(query_embed)

        hs = self.decoder(tgt, memory, memory_key_padding_mask=None,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2)
    
def sinusoid_encoding_table():
    n_position = args['chunk_size']+2
    d_hid = args['d_model']
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0).permute(1, 0, 2)

class DeviceAwareModule(nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

class cvaeEncoderInputCollator(DeviceAwareModule):
    '''
    Provides:
      * the class token embedding
      * positional encoding for a given `chunk_size` and `d_model`
      * projects actions and qpos to `d_model`
      '''
    def __init__(self):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, args['d_model']))
        self.action_proj = nn.Linear(args['action_dim'], args['d_model'])
        self.qpos_proj = nn.Linear(args['state_dim'], args['d_model'])
        self.pos_encoding = self.register_buffer('pos_encoding', sinusoid_encoding_table())
          
    def forward(self, actions, qpos, is_pad):
        bs = actions.shape[0]
        
        actions = self.action_proj(actions)       # (bs, chunk_size, d_model)
        qpos = self.qpos_proj(qpos).unsqueeze(1)  # (bs, 1, d_model)
        cls_token = self.cls_token.unsqueeze(1).repeat(bs, 1, 1) # (bs, 1, d_model)
        
        cls_qpos_actions = torch.cat([cls_token, qpos, actions], axis=1) # (bs, chunk_size+2, d_model)

        cls_qpos_is_pad = torch.zeros((bs, 2), device=self.device)
        cls_qpos_actions_is_pad = torch.cat([cls_qpos_is_pad, is_pad], axis=1)  # (bs, chunk_size+2)
        
        cls_qpos_actions = cls_qpos_actions.permute(1, 0, 2) # (chunk_size+2, bs, d_model)
        cls_qpos_actions_pos_encoding = self.pos_encoding
        return cls_qpos_actions, cls_qpos_actions_pos_encoding, cls_qpos_actions_is_pad


class cvaeEncoder(TransformerEncoder):
    def __init__(self):
        encoder_layer = TransformerEncoderLayer(args['d_model'], args['nhead'], args['dim_feedforward'],
                                        args['dropout'], "relu", args['normalize_before'])
        encoder_norm = nn.LayerNorm(d_model) if args['normalize_before'] else None
        super().__init__(encoder_layer, args['num_encoder_layers'], encoder_norm)
        
        self.cvae_encoder_input_collector = cvaeEncoderInputCollator()

    def forward(self, actions, qpos, is_pad):
        cls_qpos_actions, cls_qpos_actions_pos_encoding, cls_qpos_actions_is_pad = self.cvae_encoder_input_collector(actions, qpos, is_pad)
        encoder_output = super().forward(
            cls_qpos_actions,
            pos=cls_qpos_actions_pos_encoding,
            src_key_padding_mask=cls_qpos_actions_is_pad
        )
        style_variable_z = encoder_output[0]  # take output only for cls token
        
        return style_variable_z
    
class LatentDistributionSampler(DeviceAwareModule):
    def __init__(self):
        super().__init__()

        self.latent_var_proj =  nn.Linear(args['d_model'], args['latent_dim'] * 2)  # project latent variable to 2x latent_dim

    def forward(self, style_variable_z=None):
        if style_variable_z is not None:
            latent_distribution_parameters = self.latent_var_proj(style_variable_z)
            mu, logvar = latent_distribution_parameters.split(args['latent_dim'], dim=1)
            latent_sample = sample_from_normal_distribution(mu, logvar)
        else:
            latent_sample = torch.zeros([1, args['latent_dim']], dtype=torch.float32, device=self.device)
            mu, logvar = 0, 0
        return latent_sample, [mu, logvar]

class cvaeDecoderInputCollator(DeviceAwareModule):
    '''
    Provides:
      
      '''
    def __init__(self, vision_backbone):
        super().__init__()
        self.latent_out_proj = nn.Linear(args['latent_dim'], args['d_model'])  # project latent sample to d_model
        self.proprio_and_latent_pos = nn.Parameter(torch.randn(2, args['d_model']))
        self.robot_state_proj = nn.Linear(args['state_dim'], args['d_model'])  # project state/proprio features to d_model
        self.queries = nn.Parameter(torch.randn(args['chunk_size'], args['d_model']))
        
        self.vision_backbone = vision_backbone
        self.img_features_projector = nn.Conv2d(vision_backbone.num_channels, args['d_model'], kernel_size=1)
        
    def forward(self, latent_sample, images, qpos):
        batch_size = images.shape[0]
        
        latent_input = self.latent_out_proj(latent_sample)
        proprio_input = self.robot_state_proj(qpos)

        # image_feautres = (batch_size, d_model, feature_map_height, feature_map_width)
        # pos = like above, but 1 instead of batch_size
        image_features, pos = self.vision_backbone(images).decompose()
        
        #######
        # This is quite interesting -- feature maps with resnet 512 are 20 x 15
        # In the current formulation each "embedding" will have a collection of all pixels at a given location, say [0, 0]
        # across all feature maps. So we will get 20x15 = 300 embeddings with each containing one activation from each of the 512 channels.
        # Maybe this makes sense? It does make it simpler to ensure that we get embeddings of d_model size.
        # But one could argue that as convs operate on local patches, that it might be better to feed
        # each feature map into separate embedding. Though maybe the linear projections inside the transformer (calculating k,q,v)
        # make it not matter much/at all?
        # Might be worth trying it out at some point.
        
        image_features = image_features.flatten(2).permute(2,0,1)  # (width * height of feature map, bs, num channels == d_model)
        pos = pos.flatten(2).permute(2,0,1).repeat(1, batch_size, 1)
        
        queries = self.queries.unsqueeze(1).repeat(1, batch_size, 1)
        
        proprio_and_latent_pos = self.proprio_and_latent_pos.unsqueeze(1).repeat(1, image_features.shape[1], 1)
        pos = torch.cat([proprio_and_latent_pos, pos], axis=0)
        
        latent_and_proprio_input = torch.stack([latent_input, proprio_input], axis=1).permute(1,0,2)
        src = torch.cat([latent_and_proprio_input, image_features], axis=0)

        return src, pos, queries
    
class cvaeDecoder(Transformer):
    def __init__(self):
        
        arg_for_transformer = [
            'd_model', 'dropout', 'nhead', 'dim_feedforward', 'num_encoder_layers',
            'num_decoder_layers', 'normalize_before'
        ]
        super().__init__(**{k: args[k] for k in arg_for_transformer}) 

        vision_backbone = Backbone(args['backbone'])
        self.cvae_decoder_input_collector = cvaeDecoderInputCollator(vision_backbone)
        self.action_head = nn.Linear(args['d_model'], args['action_dim'])

    def forward(self, latent_sample, images, qpos):
        src, pos, queries = self.cvae_decoder_input_collector(latent_sample, images, qpos)
        hs = super().forward(src, pos, queries)  # hidden state from the last layer of the decoder stack
        a_hat = self.action_head(hs)
        return a_hat

def sample_from_normal_distribution(mu, logvar):
    std = logvar.div(2).exp()
    eps = std.new(std.size()).normal_()
    return mu + std * eps

class DETRVAE(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.cvaeEncoder = cvaeEncoder()
        self.latent_distribution_sampler = LatentDistributionSampler()
        self.cvaeDecoder = cvaeDecoder()

    def forward(self, qpos, images, actions=None, is_pad=None):
        """
        Dimensionalities:
          qpos: batch_size, state_dim
          image: batch_size, channel, height, width
          actions: batch_size, chunk_size, action_dim
        """
        if actions is not None:  # training
            style_variable_z = self.cvaeEncoder(actions, qpos, is_pad)    
            latent_sample, [mu, logvar] = self.latent_distribution_sampler(style_variable_z)
        else:
            latent_sample, [mu, logvar] = self.latent_distribution_sampler()
            
        a_hat = self.cvaeDecoder(latent_sample, images, qpos).squeeze(0)
        return a_hat, [mu, logvar]

class ACTPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DETRVAE().cuda()

    def __call__(self, qpos, image, actions=None, is_pad=None):
        if actions is not None: # training time
            actions = actions[:, :args['chunk_size']]  # experiment with different chunk_sizes
            is_pad = is_pad[:, :args['chunk_size']]

            a_hat, (mu, logvar) = self.model(qpos, image, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            l1 = F.l1_loss(actions, a_hat, reduction='none')
            mean_l1_masked_out_padding = (l1 * ~is_pad.unsqueeze(-1)).mean() # (l1 * ~is_pad.unsqueeze(-L1)).mean()
            loss = mean_l1_masked_out_padding + total_kld[0] * args['kl_weight']
            return loss
        else: # inference time
            a_hat, (_, _) = self.model(qpos, image) # no action, sample from prior
            return a_hat
    