# 1. Original code base
1. mjx: mahjong simulator. https://github.com/lenzlaww/MahjongAI.git
2. Tile discarding model. https://github.com/TrongTheAlpaca/mahjong_project.git

# 2. Modification based on original base code
1. We use the mjx as the mahjong simulator.
2. We use model.ipynb in Tile discarding model to train our discard model. And we combine the discard model into our RL model.

# 3. How to train and test our model
## Models
We have five different models.
1. PPO1: raw reward + random agent
2. PPO2: raw reward + shanten agent
3. PPO3: custom reward + shanten agent
4. PPO4: custom reward + shanten agent + curriculum learning
5. PPO5: custom reward + shanten agent + discard model

## How to train our models
Run files:
ppo1.ipynb
ppo2.ipynb
ppo3.ipynb
ppo4.ipynb
ppo5.ipynb

## How to test our models
Run file: 
battle.ipynb

# 4. Our trained models
1. Our all trained models location: pretrained_models/
2. Dataset for training the discard model: 


# 5. Environment

## 1. Install the virtual environment
mjx only support Linux environment, so we use WLS to build our virtual environment.
### Install WSL
```
wsl -- install -d 22.04
```

### install Ubuntu
```
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv git build-essential curl unzip
sudo apt install -y pkg-config zip g++ zlib1g-dev unzip python3 python3-pip
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg
sudo mv bazel.gpg /etc/apt/trusted.gpg.d/
echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
sudo apt update && sudo apt install -y bazel
```
### install conda
```
cd ~
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
## 2. Clone mjx
## 3. Conda Environment
```
Conda create -n mjx_env python=3.8
Conda activate mjx_env
Pip install -r requirements.txt
pip install --upgrade protobuf
find /home/{YOUR_USERNAME}/miniconda3/envs/mjx_env/lib/python3.8/site-packages/google/protobuf/internal -name "builder.py"
mkdir ~/Documents/protobuf_backup
cp /home/{YOUR_USERNAME}/miniconda3/envs/mjx_env/lib/python3.8/site-packages/google/protobuf/internal/builder.py ~/Documents/protobuf_backup/
pip install protobuf==3.19.4
cp ~/Documents/protobuf_backup/builder.py /home/{YOUR_USERNAME}/miniconda3/envs/mjx_env/lib/python3.8/site-packages/google/protobuf/internal/

pip3 install "gym>=0.25"
pip3 install torch torchvision torchaudio

```
