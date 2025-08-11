# 🎓 AI Summer School - Mastering Weights & Biases for ML Visualization

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0%2B-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Weights & Biases](https://img.shields.io/badge/Weights%20%26%20Biases-integrated-yellow)](https://wandb.ai)

A **hands-on tutorial** demonstrating the power and simplicity of **Weights & Biases (W&B)** for machine learning experiment tracking and visualization. Using Autoencoders and Variational Autoencoders on MNIST as example models, this project showcases how easy it is to set up professional-grade ML monitoring and beautiful visualizations.

## 🌟 What You'll Learn About W&B

- 📊 **Effortless Experiment Tracking**: See how simple it is to log metrics, hyperparameters, and model performance
- 🎨 **Beautiful Visualizations**: Generate stunning 2D latent space plots with just a few lines of code
- 📈 **Real-time Monitoring**: Watch your models train with live loss curves and metrics
- 🔍 **Model Introspection**: Track gradients, parameters, and model architecture automatically
- 💾 **Automatic Logging**: Save plots, checkpoints, and metadata without manual file management
- 🚀 **Zero Configuration**: Get professional ML tracking running in minutes, not hours

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/GauravR1206/AI_Summer_School.git
   cd AI_Summer_School
   ```

2. **Install dependencies**:
   ```bash
   pip install torch torchvision matplotlib wandb datasets kagglehub
   ```

3. **Set up Weights & Biases** (this is the important part! 🎯):
   ```bash
   wandb login
   ```
   
   This will open your browser to get your W&B API key. Create a free account if you don't have one - it takes 30 seconds!

### 🏃‍♂️ See W&B in Action!

Once you're set up, these simple commands will show you the magic of W&B:

#### Train Autoencoder (to see basic W&B logging)
```bash
python train_AE.py --epochs 20 --latent_dim 2
```

#### Train Variational Autoencoder (to see advanced loss tracking)
```bash
python train_VAE.py --epochs 20 --latent_dim 2
```

**🎉 That's it!** After running either command:
- Your browser will automatically open to your W&B dashboard
- You'll see real-time loss curves updating every epoch
- Beautiful 2D latent space visualizations appear every 5 epochs
- All hyperparameters, model architecture, and metrics are automatically logged

#### 🎛️ Available Arguments
```
--epochs        Number of training epochs (default: 20)
--batch_size    Batch size for training (default: 128)
--lr            Learning rate (default: 1e-3)
--latent_dim    Dimensionality of latent space (default: 2)
--project       W&B project name (default: mnist-ae/mnist-vae)
--entity        W&B entity/team name (optional)
--data_dir      Directory to store MNIST data (default: ./data)
--save_dir      Directory to save checkpoints (default: ./checkpoints)
--seed          Random seed for reproducibility (default: 42)
```

## 📁 Project Structure

```
AI_Summer_School/
├── 📄 README.md                # You are here!
├── 🧠 models.py                # Neural network architectures (AE & VAE)
├── 🏃‍♂️ train_AE.py            # Autoencoder training script
├── 🏃‍♂️ train_VAE.py           # Variational Autoencoder training script
├── 📦 pyproject.toml          # Project dependencies and metadata
├── 🔒 uv.lock                 # Dependency lock file
├── 📜 LICENSE                 # MIT License
├── 📊 data/                   # MNIST dataset (auto-downloaded)
├── 💾 checkpoints/            # Model checkpoints (created during training)
└── 📈 wandb/                  # Weights & Biases experiment logs
```

## ✨ The Magic: How Easy W&B Integration Is

See how simple it is to add professional ML tracking to any project:

### 🔧 Setup (2 lines of code!)
```python
import wandb

wandb.init(project="my-awesome-project", config={
    "learning_rate": 0.01,
    "epochs": 100,
})
```

### 📊 Logging Metrics (1 line per metric!)
```python
wandb.log({
    "loss": loss.item(),
    "accuracy": accuracy,
    "epoch": epoch
})
```

### 🎨 Beautiful Plots (W&B does the heavy lifting!)
```python
fig = plot_latent_space(model, data_loader, device)
wandb.log({"latent_space": wandb.Image(fig)})
```

## 🧠 Demo Models (Just the Vehicle for Learning W&B)

We use simple neural networks to demonstrate W&B features:

### 🔄 Autoencoder (shows basic logging)
- Simple reconstruction loss tracking
- Basic latent space visualization

### 🎲 Variational Autoencoder (shows advanced logging)
- Multiple loss components (BCE + KL divergence)
- More complex metric relationships

## 💡 Next Steps

Now that you've seen how easy W&B is, here are some exciting ways to extend your learning:

### 🚀 **Apply to Your Own Projects**
- Add `wandb.init()` to any existing training script
- Log your metrics with `wandb.log()`
- Upload plots with `wandb.Image()`
- Share beautiful experiment results with your team!

### 🔬 **Advanced W&B Experiments to Try**
- **Can you log the input and images as well like you are logging the latent space?**
  - Try logging original MNIST images alongside reconstructions
  - Compare input vs. output side-by-side in W&B
  
- **What happens to the latent space as you change the KL weight?**
  - Experiment with different β values in β-VAE
  - Watch how latent space structure changes in real-time
  
- **How about a model comparison on with different KL weights?**
  - Run multiple VAE experiments with varying KL weights
  - Use W&B's comparison tools to analyze the differences
  
- **Do you want to train other models?**
  - Try different architectures (CNNs, ResNets, Transformers)
  - Experiment with other datasets (CIFAR-10, CelebA)
  - All while maintaining the same beautiful W&B logging!


## 📝 License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Weights & Biases**: For making ML experiment tracking delightfully simple
- **PyTorch & MNIST**: Perfect tools for demonstrating W&B capabilities
- **AI Summer School participants**: Happy learning! 🎓

---

<div align="center">
  <strong>Now go make your ML experiments beautiful! ✨</strong><br>
  <em>W&B + Your Projects = Professional Results in Minutes</em>
</div>
