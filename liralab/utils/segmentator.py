import os
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import cv2
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T
import matplotlib.pyplot as plt
from monai.networks.nets import FlexibleUNet
from segmentation_models.hardsmeg.HarDNet_MSEG.lib.HarDMSEG import HarDMSEG
import logging

class Segmentator:
    def __init__(self, weights, model_type = "UnetPP"):
        """
        Available models ["UnetPP", "HarDMSEG","SeResNet50"]
        """
        available_models = ["UnetPP", "HarDMSEG","SeResNet50"]
        assert model_type in available_models, logging.error(f"Model {model_type} is unknown. Has to be one of: {available_models}.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.transform = None        

        if model_type == "UnetPP": 
            self._Segmentator__get_model_unetplusplus(weights, self.device)
            self._Segmentator__get_transform_unetplusplus()
        if model_type == "HarDMSEG":
            self._Segmentator__get_model_hardmseg(weights, self.device)
            self._Segmentator__get_transform_hardmseg()
        if model_type == "SeResNet50":
            self._Segmentator__get_model_seresnet(weights, self.device)
            self._Segmentator__get_transform_seresnet()

    def get_segmented_mask(self, input_frame: np.ndarray) -> np.ndarray:
        """
        Args:
        input_frame (np.ndarray): [H, W, C] uint8; BGR per SeResNet50.

        Returns:
        np.ndarray : [256, 256], float32, con gli indici delle classi.
            HarDMSEG/UnetPP: 0 (sfondo), 1 (target).
            SeResNet50: 0 (sfondo), 1 (giugulare), 2 (carotide).
        """

        augmented = self.transform(image=input_frame)
        input_tensor = augmented['image'].unsqueeze(0).to(self.device)  # Aggiungi dimensione batch

        with torch.inference_mode():
            output = self.model(input_tensor)

            if isinstance(self.model, HarDMSEG):
                if isinstance(output, tuple):
                    output = output[0]
                prob = torch.sigmoid(output)
                mask = (prob > 0.5).float().squeeze().cpu().numpy()
                if mask.shape != (256, 256):
                    mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
                #output = self.model(input_tensor)
            # Prendi la classe con la probabilità più alta (0 o 1)
            elif isinstance(self.model, (smp.UnetPlusPlus, FlexibleUNet)):
                # Argmax sui logits equivale ad argmax dopo la softmax.
                # Mantieni gli indici delle classi, anche per modelli multiclasse.
                mask = torch.argmax(output, dim=1)[0].cpu().numpy()
            else:
                raise TypeError(f"Modello non supportato: {type(self.model).__name__}")
        return mask.astype(np.float32, copy=False)
    
    # Private
    def __get_transform_seresnet(self):
        # Stessa pipeline di test2.py: BGR -> resize -> /255 -> (x-mean)/std.
        # A.Normalize applica già la divisione per 255: non aggiungere ToFloat.
        self.transform = A.Compose([
            A.Resize(256, 256, interpolation=cv2.INTER_LINEAR),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ])
    
    def __get_transform_unetplusplus(self):
        self.transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.ToFloat(max_value=255.0),
            ToTensorV2()
        ])
    
    def __get_transform_hardmseg(self):
        self.transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __get_model_seresnet(self, weights_path, device):
        model = FlexibleUNet(in_channels=3,out_channels=2,
            backbone="efficientnet-b2",
            pretrained=False,
            is_pad=False,
            pre_conv=None,
        )
        model.segmentation_head = nn.Conv2d(16, 3, kernel_size=1)
        model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        model.to(device).eval()
        self.model = model

    def __get_model_unetplusplus(self, weights_path, device):
        """Inizializza il modello e carica i pesi addestrati."""
        print(f"Caricamento del modello dai pesi: {weights_path}")
        model = smp.UnetPlusPlus(
            encoder_name="densenet121",
            encoder_weights=None,  # Non serve scaricare ImageNet in inferenza, carichiamo i nostri
            in_channels=3,
            classes=2,             # Sfondo (0) e Giugolare (1)
            decoder_attention_type="scse"
        )
        
        # Carica i pesi
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()  # Modalità inferenza
        
        self.model = model

    def __get_model_hardmseg(self, weights_path, device):
        model = HarDMSEG()
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()  # Modalità inferenza
        self.model = model
