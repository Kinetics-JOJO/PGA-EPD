#!/bin/bash
# Install dependencies for Prompt Distillation with MONAI models

echo "Installing required dependencies..."

# Install MONAI with all optional dependencies
pip install monai[all]

# Install other required packages if not already installed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy
pip install SimpleITK
pip install nibabel
pip install tensorboard
pip install tqdm
pip install medpy

echo "Dependencies installed successfully!"

# Download pretrained weights (optional)
echo "Creating pretrained models directory..."
mkdir -p ./pretrained_models

echo "Setup complete!"
echo "You can now run training with:"
echo "  python train_prompt_distill.py --model_type swin_unetr --student_modalities T1"