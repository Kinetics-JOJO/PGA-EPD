"""
Main training script for Prompt-based Knowledge Distillation
Supports VNet, UNETR, and Swin UNETR with DualPrompt-style injection
"""

import argparse
import os
import torch
import numpy as np
import random
from typing import List

from trainer_prompt import PromptDistillationTrainer
from prompt_modules import ModalityCombination


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_modalities(modality_str: str) -> List[int]:
    """Parse modality string to indices"""
    if modality_str.lower() == 'all':
        return [0, 1, 2, 3]  # All modalities
    
    # Parse combinations like "T1+T2" or "0,1"
    if '+' in modality_str:
        # Handle named modalities
        modalities = modality_str.split('+')
        indices = []
        for m in modalities:
            m = m.strip().upper()
            if m in ModalityCombination.MODALITY_INDICES:
                indices.append(ModalityCombination.MODALITY_INDICES[m])
        return indices
    elif ',' in modality_str:
        # Handle numeric indices
        return [int(x.strip()) for x in modality_str.split(',')]
    else:
        # Single modality
        if modality_str.isdigit():
            return [int(modality_str)]
        else:
            m = modality_str.strip().upper()
            if m in ModalityCombination.MODALITY_INDICES:
                return [ModalityCombination.MODALITY_INDICES[m]]
    
    raise ValueError(f"Invalid modality string: {modality_str}")


def main():
    parser = argparse.ArgumentParser(description='Prompt-based Knowledge Distillation Training')
    
    # Model configuration
    parser.add_argument('--model_type', type=str, default='swin_unetr',
                        choices=['vnet', 'unetr', 'swin_unetr'],
                        help='Model architecture to use')
    parser.add_argument('--freeze_encoder', action='store_true', default=True,
                        help='Freeze encoder weights (use pretrained features)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    
    # Student modality configuration
    parser.add_argument('--student_modalities', type=str, default='T1',
                        help='Student modalities (e.g., "T1", "T1+T2", "0,1,2", "all")')
    
    # Prompt configuration
    parser.add_argument('--general_prompt_length', type=int, default=5,
                        help='Length of general prompts')
    parser.add_argument('--expert_prompt_length', type=int, default=5,
                        help='Length of expert prompts')
    
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--max_epoch', type=int, default=1000,
                        help='Maximum number of epochs')
    
    # Loss weights
    parser.add_argument('--seg_weight', type=float, default=1.0,
                        help='Weight for segmentation loss')
    parser.add_argument('--kd_weight', type=float, default=10.0,
                        help='Weight for knowledge distillation loss')
    parser.add_argument('--temperature', type=float, default=10.0,
                        help='Temperature for knowledge distillation')
    
    # Data configuration
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Path to data directory')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Experiment configuration
    parser.add_argument('--log_dir', type=str, default='../log/prompt_distill',
                        help='Directory for logs and checkpoints')
    parser.add_argument('--cache_dir', type=str, default='./pretrained_models',
                        help='Directory to cache pretrained models')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Resume training
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    parser.add_argument('--ckpt_path', type=str, default='',
                        help='Path to checkpoint for resuming')
    
    # GPU configuration
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device to use')
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Parse student modalities
    student_modalities = parse_modalities(args.student_modalities)
    print(f"Student will use modalities: {student_modalities}")
    print(f"Modality combination: {ModalityCombination.get_combination_key(student_modalities)}")
    
    # Create configuration dictionary
    config = {
        # Model settings
        'model_type': args.model_type,
        'freeze_encoder': args.freeze_encoder,
        'pretrained': args.pretrained,
        'num_classes': 4,  # BraTS has 4 classes (0: background, 1: ET, 2: TC, 3: WT)
        
        # Student configuration
        'student_modalities': student_modalities,
        
        # Prompt settings
        'general_prompt_length': args.general_prompt_length,
        'expert_prompt_length': args.expert_prompt_length,
        
        # Training settings
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'max_epoch': args.max_epoch,
        
        # Loss settings
        'seg_weight': args.seg_weight,
        'kd_weight': args.kd_weight,
        'temperature': args.temperature,
        
        # Data settings
        'data_dir': args.data_dir,
        'num_workers': args.num_workers,
        
        # Experiment settings
        'log_dir': os.path.join(args.log_dir, 
                                f"{args.model_type}_{ModalityCombination.get_combination_key(student_modalities)}"),
        'cache_dir': args.cache_dir,
        'seed': args.seed,
        
        # Resume settings
        'resume': args.resume,
        'ckpt_path': args.ckpt_path,
    }
    
    # Create trainer
    print("\n" + "="*50)
    print("Initializing Prompt Distillation Trainer")
    print("="*50)
    
    trainer = PromptDistillationTrainer(config)
    
    # Start training
    print("\n" + "="*50)
    print("Starting Training")
    print("="*50)
    
    trainer.train()
    
    print("\nTraining completed successfully!")


if __name__ == '__main__':
    main()