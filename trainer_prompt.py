"""
Prompt-based Knowledge Distillation Trainer
"""

import os
import sys
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Optional, Tuple

from networks_monai import ModelRegistry
from prompt_modules import DualPromptModule, PromptedTransformerWrapper, ModalityCombination
from datasets import BraTS
from loss import DiceCeLoss, softmax_kl_loss
from evaluate import eval_one_dice, test_single_case
from utils import create_if_not


CROP_SIZE = (96, 128, 128)
STRIDE = tuple([x // 2 for x in list(CROP_SIZE)])


class PromptDistillationTrainer(nn.Module):
    """
    Trainer for Prompt-based Knowledge Distillation
    Teacher uses all modalities, Student uses subset with prompts
    """
    
    def __init__(self, cfg: Dict):
        super().__init__()
        
        print("------Prompt Distillation Configs------")
        for k, v in cfg.items():
            print(f"{k}: {v}")
        print("----------------------------------------")
        
        self.cfg = cfg
        self.num_cls = cfg.get("num_classes", 4)
        self.lr = cfg.get("lr", 0.001)
        self.max_epoch = cfg.get("max_epoch", 1000)
        self.model_type = cfg.get("model_type", "swin_unetr")  # vnet, unetr, swin_unetr
        self.freeze_encoder = cfg.get("freeze_encoder", True)
        
        # KD settings
        self.T = cfg.get("temperature", 10)
        self.kd_weight = cfg.get("kd_weight", 10)
        self.seg_weight = cfg.get("seg_weight", 1.0)
        
        # Prompt settings
        self.general_prompt_length = cfg.get("general_prompt_length", 5)
        self.expert_prompt_length = cfg.get("expert_prompt_length", 5)
        self.num_transformer_layers = self._get_num_layers()
        
        # Student modality configuration
        self.student_modalities = cfg.get("student_modalities", [0])  # Default to T1 only
        self.student_modality_key = ModalityCombination.get_combination_key(self.student_modalities)
        
        # Initialize models
        self._init_models()
        
        # Initialize optimizer (only for trainable parameters)
        self._init_optimizer()
        
        # Initialize data loaders
        self._init_dataloaders()
        
        # Loss functions
        self.dice_ce_loss = DiceCeLoss(self.num_cls)
        
        # Logging
        self._init_logging()
        
    def _get_num_layers(self) -> int:
        """Get number of transformer layers based on model type"""
        if self.model_type == 'unetr':
            return 12  # Standard ViT has 12 layers
        elif self.model_type == 'swin_unetr':
            return 4  # Swin Transformer typically has 4 stages
        else:
            return 0  # VNet doesn't have transformer layers
    
    def _init_models(self):
        """Initialize teacher and student models with prompts"""
        
        # Teacher model (uses all 4 modalities)
        print("Initializing teacher model...")
        self.teacher_model = ModelRegistry.get_model(
            model_name=self.model_type,
            in_channels=4,  # All modalities
            out_channels=self.num_cls,
            pretrained=True,
            freeze_encoder=self.freeze_encoder,
            cache_dir=self.cfg.get("cache_dir", "./pretrained_models")
        )
        self.teacher_model.cuda()
        self.teacher_model.eval()  # Teacher is always in eval mode
        
        # Freeze teacher completely
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        # Student model (uses subset of modalities)
        print("Initializing student model...")
        num_student_channels = len(self.student_modalities)
        self.student_model = ModelRegistry.get_model(
            model_name=self.model_type,
            in_channels=num_student_channels,
            out_channels=self.num_cls,
            pretrained=True,
            freeze_encoder=self.freeze_encoder,
            cache_dir=self.cfg.get("cache_dir", "./pretrained_models")
        )
        
        # Initialize prompt module if using transformer-based model
        if self.model_type in ['unetr', 'swin_unetr']:
            print("Initializing prompt module...")
            
            # Determine embedding dimension based on model
            if self.model_type == 'unetr':
                embedding_dim = 768  # ViT-Base
            else:  # swin_unetr
                embedding_dim = 48 * 8  # Feature size * 8 for last stage
            
            self.prompt_module = DualPromptModule(
                num_layers=self.num_transformer_layers,
                embedding_dim=embedding_dim,
                general_prompt_length=self.general_prompt_length,
                expert_prompt_length=self.expert_prompt_length,
                general_layers=list(range(self.num_transformer_layers // 2)),
                expert_layers=list(range(self.num_transformer_layers // 2, self.num_transformer_layers))
            )
            
            # Wrap student model with prompt injection
            self.student_model = PromptedTransformerWrapper(
                model=self.student_model,
                prompt_module=self.prompt_module,
                model_type=self.model_type
            )
        
        self.student_model.cuda()
        
        # Freeze encoder if specified
        if self.freeze_encoder and hasattr(self.student_model, 'freeze_encoder'):
            self.student_model.freeze_encoder()
            print("Froze student encoder")
    
    def _init_optimizer(self):
        """Initialize optimizer for trainable parameters only"""
        trainable_params = []
        
        # Add student model trainable parameters
        for name, param in self.student_model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
                print(f"Trainable: {name}")
        
        print(f"Total trainable parameters: {len(trainable_params)}")
        
        self.optimizer = torch.optim.Adam(
            trainable_params,
            lr=self.lr,
            weight_decay=self.cfg.get("weight_decay", 1e-5)
        )
    
    def _init_dataloaders(self):
        """Initialize training and validation data loaders"""
        # Training dataset
        train_dataset = BraTS(self.cfg.get("data_dir", "../data"), crop_size=CROP_SIZE)
        print(f"Training set includes {len(train_dataset)} samples")
        
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.cfg.get("batch_size", 4),
            shuffle=True,
            num_workers=self.cfg.get("num_workers", 4),
            pin_memory=True
        )
        
        # Validation list
        val_list = []
        val_list_path = os.path.join(self.cfg.get("data_dir", "../data"), "val_list.txt")
        with open(val_list_path, 'r') as f:
            for line in f:
                val_list.append(line.strip())
        
        self.val_list = [
            os.path.join(self.cfg.get("data_dir", "../data"), "brats2018", f"{x}.npy")
            for x in val_list
        ]
        print(f"Validation set includes {len(self.val_list)} samples")
    
    def _init_logging(self):
        """Initialize logging and tensorboard"""
        snapshot_path = self.cfg.get("log_dir", "../log/prompt_distill")
        create_if_not(snapshot_path)
        
        self.save_model_path = os.path.join(snapshot_path, "model")
        create_if_not(self.save_model_path)
        
        # Setup logging
        logging.basicConfig(
            filename=os.path.join(snapshot_path, "log.txt"),
            level=logging.INFO,
            format="[%(asctime)s.%(msecs)03d] %(message)s",
            datefmt="%H:%M:%S"
        )
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        
        # Tensorboard writer
        self.writer = SummaryWriter(os.path.join(snapshot_path, "tensorboard"))
        
        # Training state
        self.iter_num = 0
        self.start_epoch = 0
        self.best_epoch = 0
        self.best_dice = 0
        self.best_wt = 0
        self.best_co = 0
        self.best_ec = 0
        
        # Load checkpoint if resuming
        if self.cfg.get("resume", False) and self.cfg.get("ckpt_path"):
            self._load_checkpoint(self.cfg["ckpt_path"])
    
    def _load_checkpoint(self, ckpt_path: str):
        """Load checkpoint for resuming training"""
        logging.info(f"Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path)
        
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.best_dice = ckpt.get("best_dice", 0)
        self.best_epoch = ckpt.get("best_epoch", 0)
        
        # Load model state
        if "student_state_dict" in ckpt:
            self.student_model.load_state_dict(ckpt["student_state_dict"])
        
        # Load optimizer state
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        
        logging.info(f"Resumed from epoch {self.start_epoch}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through student model"""
        if hasattr(self.student_model, 'forward'):
            # If using prompted wrapper
            return self.student_model(x, self.student_modalities)
        else:
            # Standard forward
            return self.student_model(x)
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.student_model.train()
        epoch_loss = 0
        epoch_seg_loss = 0
        epoch_kd_loss = 0
        
        time_start = time.time()
        
        for idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.float().cuda(), labels.cuda()
            
            # Extract student modalities
            student_input = images[:, self.student_modalities]
            
            # Teacher forward (all modalities)
            with torch.no_grad():
                teacher_features, teacher_logits = self.teacher_model(images)
            
            # Student forward (subset modalities with prompts)
            student_features, student_logits = self.forward(student_input)
            
            # Segmentation loss
            dice_loss, ce_loss, seg_loss = self.dice_ce_loss(student_logits, labels)
            
            # Knowledge distillation loss
            kd_loss = softmax_kl_loss(
                student_logits / self.T,
                teacher_logits / self.T
            ).mean()
            
            # Total loss
            loss = self.seg_weight * seg_loss + self.kd_weight * kd_loss
            
            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Logging
            epoch_loss += loss.item()
            epoch_seg_loss += seg_loss.item()
            epoch_kd_loss += kd_loss.item()
            
            self.iter_num += 1
            
            # Tensorboard logging
            self.writer.add_scalar("train/loss", loss, self.iter_num)
            self.writer.add_scalar("train/seg_loss", seg_loss, self.iter_num)
            self.writer.add_scalar("train/kd_loss", kd_loss, self.iter_num)
            self.writer.add_scalar("train/dice_loss", dice_loss, self.iter_num)
            self.writer.add_scalar("train/ce_loss", ce_loss, self.iter_num)
            
            if idx % 10 == 0:
                logging.info(
                    f"Epoch [{epoch}/{self.max_epoch}], Iter [{idx}/{len(self.train_loader)}], "
                    f"Loss: {loss:.4f}, Seg: {seg_loss:.4f}, KD: {kd_loss:.4f}"
                )
        
        time_end = time.time()
        epoch_time = (time_end - time_start) / 60
        
        avg_loss = epoch_loss / len(self.train_loader)
        avg_seg_loss = epoch_seg_loss / len(self.train_loader)
        avg_kd_loss = epoch_kd_loss / len(self.train_loader)
        
        logging.info(
            f"Epoch {epoch} training time: {epoch_time:.2f} minutes, "
            f"Avg Loss: {avg_loss:.4f}, Avg Seg: {avg_seg_loss:.4f}, Avg KD: {avg_kd_loss:.4f}"
        )
        
        return avg_loss
    
    def validate(self, epoch: int):
        """Validate model performance"""
        self.student_model.eval()
        
        dice_all_wt = []
        dice_all_co = []
        dice_all_ec = []
        dice_all_mean = []
        
        logging.info(f"Starting validation for epoch {epoch}")
        time_start = time.time()
        
        with torch.no_grad():
            for idx, val_path in enumerate(self.val_list):
                data = np.load(val_path)
                image = data[0:4]
                label = data[4]
                
                # Extract student modalities
                student_image = image[self.student_modalities]
                
                # Predict using student model
                if hasattr(self.student_model, 'model'):
                    # If using wrapper, access underlying model for inference
                    predict, _ = test_single_case(
                        self.student_model.model,
                        student_image,
                        STRIDE,
                        CROP_SIZE,
                        self.num_cls
                    )
                else:
                    predict, _ = test_single_case(
                        self.student_model,
                        student_image,
                        STRIDE,
                        CROP_SIZE,
                        self.num_cls
                    )
                
                # Calculate dice scores
                dice_wt, dice_co, dice_ec, dice_mean = eval_one_dice(predict, label)
                dice_all_wt.append(dice_wt)
                dice_all_co.append(dice_co)
                dice_all_ec.append(dice_ec)
                dice_all_mean.append(dice_mean)
                
                if idx % 5 == 0:
                    logging.info(f"Sample [{idx}/{len(self.val_list)}], Dice: {dice_mean:.4f}")
        
        time_end = time.time()
        val_time = (time_end - time_start) / 60
        
        # Calculate mean scores
        dice_wt_mean = np.mean(dice_all_wt)
        dice_co_mean = np.mean(dice_all_co)
        dice_ec_mean = np.mean(dice_all_ec)
        dice_mean = np.mean(dice_all_mean)
        
        logging.info(
            f"Epoch {epoch} validation time: {val_time:.2f} minutes\n"
            f"Dice scores - WT: {dice_wt_mean:.4f}, TC: {dice_co_mean:.4f}, "
            f"ET: {dice_ec_mean:.4f}, Mean: {dice_mean:.4f}"
        )
        
        # Tensorboard logging
        self.writer.add_scalar("val/dice_wt", dice_wt_mean, epoch)
        self.writer.add_scalar("val/dice_co", dice_co_mean, epoch)
        self.writer.add_scalar("val/dice_ec", dice_ec_mean, epoch)
        self.writer.add_scalar("val/dice_mean", dice_mean, epoch)
        
        # Save best model
        if dice_mean > self.best_dice:
            self.best_dice = dice_mean
            self.best_epoch = epoch
            self.best_wt = dice_wt_mean
            self.best_co = dice_co_mean
            self.best_ec = dice_ec_mean
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best=True)
            logging.info(f"New best model saved with dice: {dice_mean:.4f}")
        
        return dice_mean
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_type": self.model_type,
            "student_modalities": self.student_modalities,
            "student_state_dict": self.student_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_dice": self.best_dice,
            "best_epoch": self.best_epoch,
            "config": self.cfg
        }
        
        if is_best:
            save_path = os.path.join(self.save_model_path, "best_model.pth")
        else:
            save_path = os.path.join(self.save_model_path, f"checkpoint_{epoch}.pth")
        
        torch.save(checkpoint, save_path)
        logging.info(f"Checkpoint saved to {save_path}")
    
    def train(self):
        """Main training loop"""
        logging.info("Starting Prompt Distillation Training")
        logging.info(f"Model: {self.model_type}, Student modalities: {self.student_modality_key}")
        
        train_start_time = time.time()
        
        for epoch in range(self.start_epoch, self.max_epoch):
            # Adjust learning rate
            current_lr = self.lr * (1.0 - epoch / self.max_epoch) ** 0.9
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Save checkpoint periodically
            if epoch % 25 == 0:
                self.save_checkpoint(epoch)
            
            # Validate after warmup
            if epoch >= self.max_epoch // 4:
                val_dice = self.validate(epoch)
            
            logging.info(f"Epoch {epoch} completed. Best dice so far: {self.best_dice:.4f}")
        
        # Training completed
        train_end_time = time.time()
        total_time = (train_end_time - train_start_time) / 3600
        
        self.writer.close()
        
        logging.info("=" * 50)
        logging.info("Training completed!")
        logging.info(f"Total training time: {total_time:.2f} hours")
        logging.info(f"Best epoch: {self.best_epoch}, Best dice: {self.best_dice:.4f}")
        logging.info(f"Best scores - WT: {self.best_wt:.4f}, TC: {self.best_co:.4f}, ET: {self.best_ec:.4f}")
        logging.info("=" * 50)