# PixelForge-GAN-based-Image Super Resolution
----------------------------------------------------------------------------------------------------------------------------------

An advanced deep learning project that implements **image super-resolution** using Generative Adversarial Networks (GANs).
This model enhances low-resolution images to create high-quality, detailed outputs using state-of-the-art deep learning techniques.

![image](https://github.com/user-attachments/assets/ac0cc12d-8073-4a37-8b0c-2261e8f12738)


## Project : ML hackathon 

**Nexus :  PS-1 (Computer Vision)**  
passionately trained, designed and developed by
**Department of Materials Engineering, IIT Jammu**

## Application Features
- **GAN-based Architecture**: Advanced generator-discriminator architecture for realistic image enhancement.
- **Multiple Loss Functions**: Combines adversarial, content, and perceptual (VGG) losses for optimal results.
- **Robust Image Processing**: Handles various image formats and sizes with automatic preprocessing.
- **Quality Metrics**: Evaluation with PSNR, SSIM, and normalized PSNR.
- **Web Interface**: Flask-based application for easy image enhancement.
- **GPU Acceleration**: Full CUDA support for faster processing.
- **Model Checkpointing**: Regular saving of model states for training continuity.
  
## Prerequisites
- **Python 3.8+**
- **CUDA-capable GPU** (recommended)
- **16GB+ RAM**

### Required Python Packages
```plaintext
torch>=1.8.0
torchvision>=0.9.0
numpy>=1.19.2
Pillow>=8.0.0
jupyter>=1.0.0
```

# Architecture Overview
------------------------
# Generator Network
- Initial Layer: 9x9 convolutional layer with PReLU activation.
- Residual Blocks:  16 blocks with dual 3x3 convolutional layers, batch normalization, and PReLU activation.
- Includes skip connections for efficient learning.
- Upsampling: Custom blocks with pixel shuffle for 2x resolution increase.
- Final Layer: 9x9 convolutional layer with Tanh activation.
# Discriminator Network
- Convolutional Layers: Multiple layers with increasing channels (64 to 512).
- Activation: LeakyReLU with 0.2 negative slope.
- Batch Normalization: Applied after each convolutional layer.
- Dense Layers: Final layers for classification (512 → 1024 → 1).
- 
# Challenges and Solutions
- Training Stability: Solved with hyperparameter tuning and pre-trained VGG for perceptual loss.
- Output Quality: Improved with deeper generators and residual blocks.
- Training Time: Reduced using GPUs and regular checkpointing.
- Balancing GAN Training: Handled by adjusting generator and discriminator loss weights.
- Overfitting: Mitigated with data augmentation (flipping, rotating, cropping)
# Quality Metrics
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- Normalized PSNR
- Estimated Mean Opinion Score (MOS)
# Results achieved via traininig 
Average SSIM: 0.9003
Validation PSNR: 28.99
Validation SSIM: 0.9009
# Future Enhancements
- Additional datasets for improved performance.
- Integration of user-defined enhancement options.
- Real-time enhancement capabilities.

# Contributors 
Nexus – PS-1 Team MT dept., IIT Jammu
