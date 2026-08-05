# 🎓 Thrive AI - Demo setup

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0%2B-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Weights & Biases](https://img.shields.io/badge/Weights%20%26%20Biases-integrated-yellow)](https://wandb.ai)

**Thrive AI** demonstrating the power and simplicity of **setting up and using remote GPU servers** for machine learning experiment tracking and visualization using Autoencoders.

## 🚀 Getting Started

Follow these steps to set up Weights & Biases, connect to Delta, and start training.

> **Before you start:** you need a working NCSA Delta login (ACCESS ID, NCSA username/password, and Duo). If you have not done that yet, complete the Participant HOW-TO first — Steps 1–5 there, ending with a successful `nvidia-smi` on a GPU node.

### Step 1: Create a Weights & Biases Account

1. Go to [https://wandb.ai](https://wandb.ai).
2. Click **Login** in the top-right corner of the page.
3. Select **Sign in with Google** and sign in with your Google account.
4. After signing in, you will see a screen asking you to select your role. Select **Academic**.

![W&B account creation screen — select Academic](Wandb_create.png)

5. Once you select Academic, you will land on your W&B settings page. Click **Generate API Key**, then copy the API key and save it somewhere safe on your device.

![W&B settings page — generate and copy your API key](Wandb_final.png)

> **Warning:** Do not share your API key with anyone. Do not commit it to GitHub, post it publicly, or include it in any shared files. This key is for your eyes and your eyes only.

### Step 2: Connect to Delta

Open a terminal and SSH into the Delta cluster. Replace `NCSA_USERNAME` with your actual NCSA username:

```bash
ssh NCSA_USERNAME@login.delta.ncsa.illinois.edu
```

Type your NCSA password (nothing appears as you type — that is normal), then approve the Duo push on your phone.

### Step 3: Install uv

Install [uv](https://docs.astral.sh/uv/), a fast Python package manager:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then make `uv` available in your current session and confirm it works:

```bash
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

You should see a version number such as `uv 0.9.x`. **Do not continue until this prints a version** — if it does not, see Troubleshooting below.

### Step 4: Set up your workspace

Delta has two storage areas you care about today:

| Area | Path | Use it for |
| --- | --- | --- |
| **HOME** | `/u/$USER` | Code and your Python environment. 100 GB and 750,000 files per user. |
| **WORK** | `/work/hdd/<project>/$USER` | Datasets, checkpoints, logs — anything a job reads or writes. |

Home is **not** meant for job I/O, so we keep the repository and the virtual environment in home and send all data and outputs to work. Set that up once:

```bash
accounts                              # confirm your project code (e.g. bfep)
export PROJ=/work/hdd/bfep/$USER      # edit if your project code differs
mkdir -p $PROJ/data $PROJ/checkpoints $PROJ/wandb
export WANDB_DIR=$PROJ/wandb
```

### Step 5: Clone the Repository and Install Dependencies

Clone this repository using HTTPS (no GitHub account needed — it is public) and install the dependencies with `uv`:

```bash
cd $HOME
git clone https://github.com/GauravR1206/Eyes-on-your-model.git
cd Eyes-on-your-model
uv sync -p 3.12
```

`-p 3.12` pins the Python version. PyTorch 2.8 does not publish wheels for Python 3.14, so without the pin `uv` may pick an interpreter that cannot install torch.

> This downloads PyTorch and the CUDA libraries — roughly 4–6 GB and a few minutes on a good connection. It only happens once.

### Step 6: Log in to Weights & Biases

Run the following and paste your API key when prompted:

```bash
uv run wandb login
```

Use `uv run` here — `wandb` lives inside the project's virtual environment, so plain `wandb login` will report `command not found` until the environment is activated in Step 7.

> **Your key will not appear as you paste it.** That is normal — paste and press Enter. The key is saved to `~/.netrc`, so you only do this once; it will still work after you move to a GPU node.

### Step 7: Request a GPU

Submit an interactive GPU job on Delta:

```bash
srun -A bfep-delta-gpu --partition=gpuA40x4-interactive \
     --nodes=1 --gpus-per-node=1 --cpus-per-task=4 --mem=16g \
     --time=00:20:00 --pty bash
```

Wait until you are assigned a GPU — your prompt will change to something like `[NCSA_USERNAME@gpua045 ...]$`. Then activate your virtual environment:

```bash
source .venv/bin/activate
```

### Step 8: Train the Autoencoder

Run the autoencoder training script, writing data and checkpoints to your work directory:

```bash
python train_AE.py --epochs 20 --latent_dim 2 \
    --data_dir $PROJ/data --save_dir $PROJ/checkpoints
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

## 🩺 Troubleshooting

| Symptom | What it means / what to do |
| --- | --- |
| `source $HOME/.local/bin/env: No such file or directory` | The installer did not write that helper script. Use `export PATH="$HOME/.local/bin:$PATH"` instead — that is all the script does. |
| `uv: command not found` after Step 3 | Same cause. Run the `export PATH` line, then `uv --version`. If still missing, re-run the `curl` installer and read its output for errors. |
| `Username for 'https://github.com':` during `git clone` | The URL is wrong or the repository is private. Press Ctrl+C. Public repositories never ask for credentials — check the clone URL against Step 5. |
| `torch ... doesn't have a source distribution or wheel for the current platform` | `uv` selected Python 3.14. Run `rm -rf .venv` and re-run `uv sync -p 3.12`. |
| `wandb: command not found` | `wandb` is in the project environment. Use `uv run wandb login`, or activate first with `source .venv/bin/activate`. |
| `No pyproject.toml found` from `uv sync` | You are not inside the repository directory. Run `cd $HOME/Eyes-on-your-model` first. |
| `srun` sits at "queued and waiting" | The GPU queue is busy; it usually starts within a minute or two. Ctrl+C and retry if it stalls. |
| `Invalid account` on `srun` | Check the account string is exactly `-A bfep-delta-gpu`. |

## 📝 License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.
