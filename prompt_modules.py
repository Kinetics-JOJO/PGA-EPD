"""
Prompt modules for prompt-based knowledge distillation
Inspired by DualPrompt for continual learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import itertools
import numpy as np


class ModalityCombination:
    """Generate all possible modality combinations"""
    
    MODALITIES = ['T1', 'T2', 'T1ce', 'FLAIR']
    MODALITY_INDICES = {'T1': 0, 'T2': 1, 'T1ce': 2, 'FLAIR': 3}
    
    @classmethod
    def get_all_combinations(cls):
        """Get all possible non-empty combinations of modalities"""
        combinations = []
        for r in range(1, len(cls.MODALITIES) + 1):
            for combo in itertools.combinations(cls.MODALITIES, r):
                combinations.append(combo)
        return combinations
    
    @classmethod
    def get_combination_key(cls, modality_indices: List[int]) -> str:
        """Convert modality indices to combination key"""
        modalities = [cls.MODALITIES[i] for i in sorted(modality_indices)]
        return '+'.join(modalities)
    
    @classmethod
    def get_indices_from_key(cls, key: str) -> List[int]:
        """Convert combination key to modality indices"""
        modalities = key.split('+')
        return [cls.MODALITY_INDICES[m] for m in modalities]


class PromptPool(nn.Module):
    """Pool of learnable prompts"""
    
    def __init__(
        self,
        num_prompts: int,
        prompt_length: int,
        embedding_dim: int,
        prompt_init: str = 'uniform',
        prompt_key_init: str = 'uniform'
    ):
        super().__init__()
        self.num_prompts = num_prompts
        self.prompt_length = prompt_length
        self.embedding_dim = embedding_dim
        
        # Initialize prompt embeddings
        self.prompts = nn.Parameter(
            torch.zeros(num_prompts, prompt_length, embedding_dim)
        )
        
        # Initialize prompt keys for selection
        self.prompt_keys = nn.Parameter(
            torch.zeros(num_prompts, embedding_dim)
        )
        
        # Initialize parameters
        self._init_prompts(prompt_init)
        self._init_keys(prompt_key_init)
        
    def _init_prompts(self, init_type):
        """Initialize prompt embeddings"""
        if init_type == 'uniform':
            nn.init.uniform_(self.prompts, -0.02, 0.02)
        elif init_type == 'normal':
            nn.init.normal_(self.prompts, std=0.02)
        elif init_type == 'xavier':
            nn.init.xavier_uniform_(self.prompts)
            
    def _init_keys(self, init_type):
        """Initialize prompt keys"""
        if init_type == 'uniform':
            nn.init.uniform_(self.prompt_keys, -1, 1)
        elif init_type == 'normal':
            nn.init.normal_(self.prompt_keys, std=1.0)
        elif init_type == 'orthogonal':
            nn.init.orthogonal_(self.prompt_keys)
            
    def forward(self, query_features: torch.Tensor, top_k: int = 1) -> torch.Tensor:
        """
        Select prompts based on query features
        
        Args:
            query_features: Features to match against keys [B, D]
            top_k: Number of prompts to select
            
        Returns:
            Selected prompts [B, top_k * prompt_length, D]
        """
        B = query_features.shape[0]
        
        # Compute similarity between query and keys
        query_norm = F.normalize(query_features, dim=-1)  # [B, D]
        key_norm = F.normalize(self.prompt_keys, dim=-1)  # [N, D]
        
        similarity = torch.matmul(query_norm, key_norm.T)  # [B, N]
        
        # Select top-k prompts
        _, indices = torch.topk(similarity, top_k, dim=-1)  # [B, top_k]
        
        # Gather selected prompts
        selected_prompts = []
        for b in range(B):
            batch_prompts = []
            for k in range(top_k):
                idx = indices[b, k]
                batch_prompts.append(self.prompts[idx])
            selected_prompts.append(torch.cat(batch_prompts, dim=0))
            
        return torch.stack(selected_prompts)  # [B, top_k * prompt_length, D]


class DualPromptModule(nn.Module):
    """
    Dual Prompt module with general and expert prompts
    General prompts: shared across all modality combinations (early layers)
    Expert prompts: specific to modality combinations (later layers)
    """
    
    def __init__(
        self,
        num_layers: int = 12,
        embedding_dim: int = 768,
        general_prompt_length: int = 5,
        expert_prompt_length: int = 5,
        general_layers: List[int] = None,
        expert_layers: List[int] = None
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        
        # Define which layers get which prompts
        if general_layers is None:
            # First half layers get general prompts
            self.general_layers = list(range(num_layers // 2))
        else:
            self.general_layers = general_layers
            
        if expert_layers is None:
            # Second half layers get expert prompts
            self.expert_layers = list(range(num_layers // 2, num_layers))
        else:
            self.expert_layers = expert_layers
        
        # Create general prompt pool (shared across all modalities)
        self.general_prompt_pool = PromptPool(
            num_prompts=1,  # Single shared general prompt
            prompt_length=general_prompt_length,
            embedding_dim=embedding_dim
        )
        
        # Create expert prompt pools for each modality combination
        self.expert_prompt_pools = nn.ModuleDict()
        combinations = ModalityCombination.get_all_combinations()
        
        for combo in combinations:
            key = '+'.join(combo)
            self.expert_prompt_pools[key] = PromptPool(
                num_prompts=1,  # One expert prompt per combination
                prompt_length=expert_prompt_length,
                embedding_dim=embedding_dim
            )
        
        self.general_prompt_length = general_prompt_length
        self.expert_prompt_length = expert_prompt_length
        
    def get_prompts_for_layer(
        self,
        layer_idx: int,
        modality_key: str,
        query_features: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """
        Get prompts for a specific layer and modality combination
        
        Args:
            layer_idx: Index of the transformer layer
            modality_key: Key representing modality combination (e.g., 'T1+T2')
            query_features: Features for prompt selection [B, D]
            
        Returns:
            Prompts for this layer [B, prompt_length, D] or None
        """
        B = query_features.shape[0] if query_features is not None else 1
        
        if layer_idx in self.general_layers:
            # Return general prompts
            if query_features is not None:
                prompts = self.general_prompt_pool(query_features, top_k=1)
            else:
                # Return the single general prompt repeated for batch
                prompts = self.general_prompt_pool.prompts[0].unsqueeze(0).repeat(B, 1, 1)
            return prompts
            
        elif layer_idx in self.expert_layers:
            # Return expert prompts for this modality combination
            if modality_key in self.expert_prompt_pools:
                if query_features is not None:
                    prompts = self.expert_prompt_pools[modality_key](query_features, top_k=1)
                else:
                    # Return the expert prompt for this combination
                    prompts = self.expert_prompt_pools[modality_key].prompts[0].unsqueeze(0).repeat(B, 1, 1)
                return prompts
            else:
                # Fallback to general prompt if combination not found
                if query_features is not None:
                    prompts = self.general_prompt_pool(query_features, top_k=1)
                else:
                    prompts = self.general_prompt_pool.prompts[0].unsqueeze(0).repeat(B, 1, 1)
                return prompts
        
        return None
    
    def forward(
        self,
        features: torch.Tensor,
        layer_idx: int,
        modality_indices: List[int]
    ) -> torch.Tensor:
        """
        Add prompts to features at a specific layer
        
        Args:
            features: Input features [B, N, D]
            layer_idx: Current layer index
            modality_indices: Indices of active modalities
            
        Returns:
            Features with prompts prepended [B, N + prompt_length, D]
        """
        modality_key = ModalityCombination.get_combination_key(modality_indices)
        
        # Get prompts for this layer
        prompts = self.get_prompts_for_layer(
            layer_idx,
            modality_key,
            query_features=features.mean(dim=1)  # Use mean pooled features as query
        )
        
        if prompts is not None:
            # Prepend prompts to features
            features = torch.cat([prompts, features], dim=1)
        
        return features


class PromptedTransformerWrapper(nn.Module):
    """
    Wrapper to add prompts to transformer-based models
    Works with UNETR and Swin UNETR
    """
    
    def __init__(
        self,
        model: nn.Module,
        prompt_module: DualPromptModule,
        model_type: str = 'unetr'  # 'unetr' or 'swin_unetr'
    ):
        super().__init__()
        self.model = model
        self.prompt_module = prompt_module
        self.model_type = model_type
        
    def forward(self, x: torch.Tensor, modality_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with prompt injection
        
        Args:
            x: Input tensor [B, C, H, W, D]
            modality_indices: Indices of active modalities
            
        Returns:
            features: Encoder features
            logits: Segmentation output
        """
        if self.model_type == 'unetr':
            return self._forward_unetr(x, modality_indices)
        elif self.model_type == 'swin_unetr':
            return self._forward_swin_unetr(x, modality_indices)
        else:
            # Fallback to standard forward
            return self.model(x)
    
    def _forward_unetr(self, x: torch.Tensor, modality_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for UNETR with prompt injection"""
        # This would need to be implemented with hooks to inject prompts
        # at different transformer layers
        # For now, return standard forward
        return self.model(x)
    
    def _forward_swin_unetr(self, x: torch.Tensor, modality_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for Swin UNETR with prompt injection"""
        # This would need to be implemented with hooks to inject prompts
        # at different transformer layers
        # For now, return standard forward
        return self.model(x)
    
    def freeze_encoder(self):
        """Freeze encoder parameters"""
        if hasattr(self.model, 'freeze_encoder'):
            self.model.freeze_encoder()
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters"""
        if hasattr(self.model, 'unfreeze_encoder'):
            self.model.unfreeze_encoder()