#!/bin/bash
# Example training scripts for different configurations

# Make sure dependencies are installed
# bash install_dependencies.sh

echo "======================================"
echo "Prompt Distillation Training Examples"
echo "======================================"

# Example 1: Swin UNETR with T1 only (single modality)
echo "Training Swin UNETR with T1 modality only..."
python train_prompt_distill.py \
    --model_type swin_unetr \
    --student_modalities T1 \
    --freeze_encoder \
    --pretrained \
    --batch_size 4 \
    --lr 0.001 \
    --max_epoch 1000 \
    --kd_weight 10.0 \
    --seg_weight 1.0 \
    --log_dir ../log/prompt_swin_t1 \
    --gpu 0

# Example 2: UNETR with T1+T2 (two modalities)
echo "Training UNETR with T1+T2 modalities..."
python train_prompt_distill.py \
    --model_type unetr \
    --student_modalities "T1+T2" \
    --freeze_encoder \
    --pretrained \
    --batch_size 4 \
    --lr 0.001 \
    --max_epoch 1000 \
    --kd_weight 10.0 \
    --seg_weight 1.0 \
    --log_dir ../log/prompt_unetr_t1t2 \
    --gpu 0

# Example 3: VNet baseline with T1ce only
echo "Training VNet with T1ce modality..."
python train_prompt_distill.py \
    --model_type vnet \
    --student_modalities T1ce \
    --batch_size 4 \
    --lr 0.001 \
    --max_epoch 1000 \
    --kd_weight 10.0 \
    --seg_weight 1.0 \
    --log_dir ../log/prompt_vnet_t1ce \
    --gpu 0

# Example 4: Swin UNETR with three modalities (T1+T2+FLAIR)
echo "Training Swin UNETR with T1+T2+FLAIR..."
python train_prompt_distill.py \
    --model_type swin_unetr \
    --student_modalities "T1+T2+FLAIR" \
    --freeze_encoder \
    --pretrained \
    --general_prompt_length 10 \
    --expert_prompt_length 10 \
    --batch_size 2 \
    --lr 0.0005 \
    --max_epoch 800 \
    --kd_weight 5.0 \
    --seg_weight 1.0 \
    --log_dir ../log/prompt_swin_t1t2flair \
    --gpu 0

# Example 5: Resume training from checkpoint
echo "Resuming training from checkpoint..."
python train_prompt_distill.py \
    --model_type swin_unetr \
    --student_modalities T1 \
    --freeze_encoder \
    --pretrained \
    --resume \
    --ckpt_path ../log/prompt_swin_t1/model/checkpoint_100.pth \
    --batch_size 4 \
    --lr 0.001 \
    --max_epoch 1000 \
    --log_dir ../log/prompt_swin_t1_resume \
    --gpu 0