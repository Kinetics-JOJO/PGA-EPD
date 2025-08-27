# Prompt-based Knowledge Distillation for Medical Image Segmentation

This is an enhanced version of the PGA-EPD project that implements **Prompt-based Knowledge Distillation** with support for multiple backbone architectures from MONAI.

## Key Features

### 1. Multiple Backbone Support
- **VNet**: Original 3D U-Net variant
- **UNETR**: Vision Transformer-based architecture
- **Swin UNETR**: Swin Transformer-based architecture with hierarchical features

### 2. Prompt Distillation Mechanism
Inspired by DualPrompt, our approach includes:
- **General Prompts**: Shared knowledge across all modality combinations (early layers)
- **Expert Prompts**: Specialized knowledge for specific modality combinations (later layers)
- **Modality-aware Injection**: Prompts are injected at different MSA (Multi-head Self-Attention) layers

### 3. Flexible Modality Combinations
Support for all possible combinations of BraTS modalities:
- Single modality: T1, T2, T1ce, or FLAIR
- Dual modalities: T1+T2, T1+T1ce, etc.
- Triple modalities: T1+T2+T1ce, etc.
- All four modalities: T1+T2+T1ce+FLAIR

### 4. Pretrained Weight Support
- Automatic download of BraTS-pretrained weights
- Frozen encoder for efficient fine-tuning
- Trainable decoder and prompt modules

## Installation

```bash
# Install dependencies
bash install_dependencies.sh

# Or manually install
pip install monai[all]
pip install torch torchvision SimpleITK nibabel tensorboard tqdm medpy
```

## Usage

### Training with Prompt Distillation

#### Basic Training Command
```bash
python train_prompt_distill.py \
    --model_type swin_unetr \
    --student_modalities T1 \
    --freeze_encoder \
    --batch_size 4 \
    --lr 0.001 \
    --max_epoch 1000 \
    --gpu 0
```

#### Advanced Options

**Model Selection:**
```bash
--model_type {vnet, unetr, swin_unetr}
```

**Modality Combinations:**
```bash
# Single modality
--student_modalities T1

# Multiple modalities (use + separator)
--student_modalities "T1+T2"
--student_modalities "T1+T2+FLAIR"

# Using indices
--student_modalities "0,1,2"  # T1, T2, T1ce

# All modalities
--student_modalities all
```

**Prompt Configuration:**
```bash
--general_prompt_length 5  # Length of general prompts
--expert_prompt_length 5   # Length of expert prompts
```

**Loss Weights:**
```bash
--seg_weight 1.0      # Segmentation loss weight
--kd_weight 10.0      # Knowledge distillation weight
--temperature 10.0    # KD temperature
```

### Evaluation

```bash
python evaluate_prompt.py \
    --model_path ../log/prompt_swin_t1/model/best_model.pth \
    --data_dir ../data \
    --output_path ../results/swin_t1 \
    --save_vis  # Optional: save visualization
```

### Example Training Scenarios

See `run_experiments.sh` for complete examples:

```bash
# Run all example experiments
bash run_experiments.sh
```

## Architecture Details

### Prompt Injection Process

1. **Early Layers (General Prompts)**:
   - Layers 0 to N/2-1 receive general prompts
   - Shared across all modality combinations
   - Capture fundamental image features

2. **Later Layers (Expert Prompts)**:
   - Layers N/2 to N-1 receive expert prompts
   - Specific to each modality combination
   - Capture modality-specific patterns

### Knowledge Distillation

1. **Teacher Model**:
   - Uses all 4 modalities (T1, T2, T1ce, FLAIR)
   - Pretrained on BraTS dataset
   - Frozen during training

2. **Student Model**:
   - Uses subset of modalities
   - Encoder frozen (using pretrained features)
   - Decoder and prompts are trainable

3. **Loss Function**:
   ```
   L_total = λ_seg * L_seg + λ_kd * L_kd
   ```
   - L_seg: Dice + Cross-entropy loss
   - L_kd: KL divergence between teacher and student outputs

## File Structure

```
├── networks_monai.py          # MONAI model wrappers
├── prompt_modules.py          # Prompt injection modules
├── trainer_prompt.py          # Prompt distillation trainer
├── train_prompt_distill.py    # Main training script
├── evaluate_prompt.py         # Evaluation script
├── install_dependencies.sh    # Dependency installation
├── run_experiments.sh         # Example experiments
└── README_PROMPT_DISTILL.md   # This file
```

## Key Improvements Over Original

1. **Model Flexibility**: Switch between VNet, UNETR, and Swin UNETR
2. **Prompt-based Learning**: More efficient knowledge transfer
3. **Pretrained Weights**: Better initialization from BraTS-trained models
4. **Modality Flexibility**: Support for any combination of modalities
5. **Frozen Encoder**: More efficient training with pretrained features

## Performance Tips

1. **GPU Memory**:
   - Swin UNETR requires more memory than VNet
   - Reduce batch size if encountering OOM errors
   - Consider gradient accumulation for larger effective batch sizes

2. **Training Speed**:
   - Frozen encoder significantly speeds up training
   - VNet is fastest, Swin UNETR is slowest but most accurate

3. **Hyperparameters**:
   - Start with default values
   - Adjust KD weight based on validation performance
   - Longer prompts may help for complex modality combinations

## Troubleshooting

1. **MONAI Installation Issues**:
   ```bash
   # Try installing without optional dependencies
   pip install monai
   ```

2. **CUDA Memory Issues**:
   - Reduce batch size
   - Use gradient checkpointing (if available)
   - Try smaller model (VNet instead of Swin UNETR)

3. **Pretrained Weight Download**:
   - Weights are cached in `./pretrained_models/`
   - Manual download links available in `networks_monai.py`

## Citation

If you use this code, please cite:
- Original ProtoKD paper
- MONAI framework
- Respective model papers (UNETR, Swin UNETR)

## License

This project extends the original PGA-EPD codebase with additional features.
Please refer to the original license terms.