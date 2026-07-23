import os
import glob
import time
import numpy as np
import torch
import cv2
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from segmentation_models.hardsmeg.HarDNet_MSEG.lib.HarDMSEG import HarDMSEG
import logging

class Segmentator:
    def __init__(self, weights, model_type = "UnetPP"):
        available_models = ["UnetPP", "HarDMSEG"]
        assert model_type in available_models, logging.error(f"Model {model_type} is unknown. Has to be one of: {available_models}.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.transform = None        
        
        if model_type == "UnetPP": 
            self._Segmentator_get_model_unetplusplus(weights, self.device)
            self._Segmentator__get_transform_unetplusplus()
        if model_type == "HarDMSEG":
            self._Segmentator__get_model_hardmseg(weights, self.device)
            self._Segmentator__get_transform_hardmseg()

    def get_segmented_mask(self, input_frame: np.ndarray) -> np.ndarray:
        """
        Args:
        input_frame (np.ndarray): [H, W, C] (uint8)

        Returns:
        np.ndarray : [256, 256], (flaot32) (0.0 -> 1.0)
        """

        augmented = self.transform(image=input_frame)
        input_tensor = augmented['image'].unsqueeze(0).to(self.device)  # Aggiungi dimensione batch

        with torch.no_grad():
            output = self.model(input_tensor)

            if isinstance(self.model, HarDMSEG):
                if isinstance(output, tuple):
                    output = output[0]
                prob = torch.sigmoid(output)
                mask = (prob > 0.5).float().squeeze().cpu().numpy()
                output = self.model(input_tensor)
            # Prendi la classe con la probabilità più alta (0 o 1)
            elif isinstance(self.model, smp.UnetPlusPlus):
                mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            else:
                print("Model is not a known instance")
        return mask
    
    # Private
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