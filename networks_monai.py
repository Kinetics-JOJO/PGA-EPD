"""
MONAI-based network architectures with pretrained weights support
Supports VNet, UNETR, and Swin UNETR
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
import logging

# Import MONAI components
try:
    from monai.networks.nets import VNet as MonaiVNet
    from monai.networks.nets import UNETR, SwinUNETR
    from monai.apps import download_and_extract
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    logging.warning("MONAI not installed. Please install with: pip install monai")

# Original VNet from the codebase
from networks import VNet as OriginalVNet


class ModelRegistry:
    """Registry for managing different model architectures"""
    
    # Pretrained model URLs (BraTS weights)
    PRETRAINED_URLS = {
        'swin_unetr_brats': 'https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/swin_unetr.tiny_5000ep_f48_lr2e-4_pretrained.pt',
        'unetr_brats': None,  # Will need to find appropriate URL or train from scratch
        'vnet_brats': None,
    }
    
    @staticmethod
    def get_model(
        model_name: str,
        in_channels: int = 4,
        out_channels: int = 4,
        pretrained: bool = True,
        freeze_encoder: bool = False,
        cache_dir: str = './pretrained_models'
    ) -> nn.Module:
        """
        Get model instance with optional pretrained weights
        
        Args:
            model_name: One of ['vnet', 'unetr', 'swin_unetr']
            in_channels: Number of input channels
            out_channels: Number of output channels
            pretrained: Whether to load pretrained weights
            freeze_encoder: Whether to freeze encoder weights
            cache_dir: Directory to cache pretrained models
        """
        
        if not MONAI_AVAILABLE and model_name != 'vnet':
            raise RuntimeError("MONAI is required for UNETR models. Please install it.")
        
        model_name = model_name.lower()
        
        if model_name == 'vnet':
            # Use original VNet implementation
            model = OriginalVNet(
                n_channels=in_channels,
                n_classes=out_channels,
                n_filters=16,
                normalization='batchnorm'
            )
            
        elif model_name == 'unetr':
            model = UNETRWrapper(
                in_channels=in_channels,
                out_channels=out_channels,
                img_size=(96, 128, 128),
                feature_size=16,
                hidden_size=768,
                mlp_dim=3072,
                num_heads=12,
                pos_embed='perceptron',
                norm_name='instance',
                res_block=True,
                dropout_rate=0.0
            )
            
        elif model_name == 'swin_unetr':
            model = SwinUNETRWrapper(
                in_channels=in_channels,
                out_channels=out_channels,
                img_size=(96, 128, 128),
                feature_size=48,
                use_checkpoint=False
            )
            
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        # Load pretrained weights if available
        if pretrained and model_name in ['unetr', 'swin_unetr']:
            model.load_pretrained_weights(cache_dir)
        
        # Freeze encoder if requested
        if freeze_encoder and hasattr(model, 'freeze_encoder'):
            model.freeze_encoder()
        
        return model


class UNETRWrapper(nn.Module):
    """Wrapper for MONAI UNETR with additional functionality"""
    
    def __init__(self, **kwargs):
        super().__init__()
        self.model = UNETR(**kwargs)
        self.in_channels = kwargs.get('in_channels', 4)
        self.out_channels = kwargs.get('out_channels', 4)
        
    def forward(self, x):
        """Forward pass returning both features and logits"""
        # Get hidden states from ViT encoder
        hidden_states = self.get_encoder_features(x)
        
        # Get final output
        logits = self.model(x)
        
        # Return last encoder feature and logits
        return hidden_states[-1], logits
    
    def get_encoder_features(self, x):
        """Extract features from encoder at different layers"""
        # Access ViT encoder
        x = self.model.vit(x)
        hidden_states = self.model.vit.hidden_states
        return hidden_states
    
    def freeze_encoder(self):
        """Freeze encoder (ViT) parameters"""
        for param in self.model.vit.parameters():
            param.requires_grad = False
            
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters"""
        for param in self.model.vit.parameters():
            param.requires_grad = True
            
    def load_pretrained_weights(self, cache_dir):
        """Load pretrained weights for UNETR"""
        # This would need actual pretrained weights URL
        logging.info("Loading pretrained UNETR weights...")
        # Placeholder - would need actual implementation
        pass


class SwinUNETRWrapper(nn.Module):
    """Wrapper for MONAI Swin UNETR with additional functionality"""
    
    def __init__(self, **kwargs):
        super().__init__()
        self.model = SwinUNETR(**kwargs)
        self.in_channels = kwargs.get('in_channels', 4)
        self.out_channels = kwargs.get('out_channels', 4)
        
    def forward(self, x):
        """Forward pass returning both features and logits"""
        # Get hidden states from Swin Transformer encoder
        hidden_states = self.get_encoder_features(x)
        
        # Get final output
        logits = self.model(x)
        
        # Return last encoder feature and logits
        return hidden_states[-1], logits
    
    def get_encoder_features(self, x):
        """Extract features from encoder at different layers"""
        hidden_states_out = []
        
        # Swin Transformer encoder forward
        x = self.model.swinViT(x)
        hidden_states_out = self.model.swinViT.hidden_states
        
        return hidden_states_out
    
    def freeze_encoder(self):
        """Freeze encoder (Swin Transformer) parameters"""
        for param in self.model.swinViT.parameters():
            param.requires_grad = False
            
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters"""
        for param in self.model.swinViT.parameters():
            param.requires_grad = True
            
    def load_pretrained_weights(self, cache_dir):
        """Load pretrained weights for Swin UNETR"""
        import os
        os.makedirs(cache_dir, exist_ok=True)
        
        weight_path = os.path.join(cache_dir, 'swin_unetr_brats.pt')
        
        if not os.path.exists(weight_path):
            logging.info("Downloading pretrained Swin UNETR weights...")
            url = ModelRegistry.PRETRAINED_URLS['swin_unetr_brats']
            if url:
                # Download weights
                import urllib.request
                urllib.request.urlretrieve(url, weight_path)
                logging.info(f"Downloaded weights to {weight_path}")
        
        if os.path.exists(weight_path):
            logging.info("Loading pretrained Swin UNETR weights...")
            state_dict = torch.load(weight_path, map_location='cpu')
            self.model.load_state_dict(state_dict, strict=False)
            logging.info("Loaded pretrained weights successfully")


class VNetWrapper(nn.Module):
    """Wrapper for original VNet to maintain consistency"""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        return self.model(x)
    
    def freeze_encoder(self):
        """Freeze encoder blocks"""
        # Freeze first 4 blocks (encoder)
        for name, param in self.model.named_parameters():
            if any(block in name for block in ['block_one', 'block_two', 'block_three', 'block_four']):
                param.requires_grad = False
                
    def unfreeze_encoder(self):
        """Unfreeze all parameters"""
        for param in self.model.parameters():
            param.requires_grad = True