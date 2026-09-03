# 2.2 Choose a Setup

You can read this chapter and use the included figures and rollout explorer
without installing anything.

To run the Python examples with the included videos, use the basic setup below.
To generate new rollouts, complete the basic setup and the optional GPU setup.

## Basic Setup

Clone the repository, create a virtual environment, and install the project:

```bash
git clone https://github.com/nahidalam/world_model_from_scratch
cd world_model_from_scratch
python -m venv wm_env
source wm_env/bin/activate
pip install -r requirements.txt
pip install -e .
```

This environment can load and analyze the included videos. It does not install
the Cosmos inference stack.

## Generate New Rollouts (Optional)

Generation requires Linux and an NVIDIA GPU visible to PyTorch. The verified
configuration used about 33 GB of memory on an NVIDIA A40; plan for a GPU with
at least 40 GB of memory.

On Ubuntu, install the system libraries and the verified Cosmos dependencies:

```bash
sudo apt-get install -y libxcb1 libgl1 libglib2.0-0
pip install -r requirements-chapter-02-gpu.txt
pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
```


The Cosmos-Predict checkpoint is gated. Accept its license on the
[model page](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B), then sign in:

```bash
hf auth login
```

If the default Hugging Face cache does not have enough space, choose another
location before loading the model:

```bash
export HF_HOME=/path/with/space
```
