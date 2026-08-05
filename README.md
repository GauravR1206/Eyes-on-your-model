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

## 🚀 Getting Started

Follow these steps to set up Weights & Biases, connect to Delta AI, and start training.

### Step 1: Create a Weights & Biases Account

1. Go to [https://wandb.ai](https://wandb.ai).
2. Click **Login** in the top-right corner of the page.
3. Select **Sign in with Google** and sign in with your Google account.
4. After signing in, you will see a screen asking you to select your role. Select **Academic**.

![W&B account creation screen — select Academic](Wandb_create.png)

5. Once you select Academic, you will land on your W&B settings page. Click **Generate API Key**, then copy the API key and save it somewhere safe on your device.

![W&B settings page — generate and copy your API key](Wandb_final.png)

> **Warning:** Do not share your API key with anyone. Do not commit it to GitHub, post it publicly, or include it in any shared files. This key is for your eyes and your eyes only.

### Step 2: Connect to Delta AI

Open a terminal and SSH into the Delta cluster. Replace `NCSA_USERNAME` with your actual NCSA username:

```bash
ssh NCSA_USERNAME@login.delta.ncsa.illinois.edu
```

### Step 3: Install uv

Before cloning the repository, install [uv](https://docs.astral.sh/uv/), a fast Python package manager:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, restart your shell or run the following to make `uv` available in your current session:

```bash
source $HOME/.local/bin/env
```

### Step 4: Clone the Repository and Install Dependencies

Clone this repository using HTTPS and install the dependencies with `uv`:

```bash
git clone https://github.com/GauravR1206/AI_Summer_School.git
cd AI_Summer_School
uv sync
```

### Step 5: Log in to Weights & Biases

Run the following command and paste your API key when prompted:

```bash
wandb login
```

### Step 6: Request a GPU

Submit an interactive GPU job on Delta:

```bash
srun -A bfep-delta-gpu --partition=gpuA40x4-interactive \
     --nodes=1 --gpus-per-node=1 --cpus-per-task=4 --mem=16g \
     --time=00:20:00 --pty bash
```

Wait until you are assigned a GPU. Once you have a GPU, activate your virtual environment:

```bash
source .venv/bin/activate
```

### Step 7: Train the Autoencoder

Now run the autoencoder training script:

```bash
python train_AE.py --epochs 20 --latent_dim 2
```

Once training begins, you will see a link printed in the terminal that takes you to your W&B dashboard. Open that link to monitor your model's training in real time — you'll see live loss curves, latent space visualizations, and all logged metrics.

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
