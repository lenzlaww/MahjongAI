import torch
import ast 
import mjx
import random
from typing import List
from tqdm import tqdm
from mjx.agents import RandomAgent
import json


agent = RandomAgent()
env = mjx.MjxEnv()
N = 10
for _ in tqdm(range(N)):
  obs_dict = env.reset()
  while not env.done():
      actions = {player_id: agent.act(obs)
              for player_id, obs in obs_dict.items()}
      obs_dict = env.step(actions)
  returns = env.rewards()

state = env.state()
obs = obs_dict['player_0']


roundWind = obs.round()
dealer = obs.dealer()
POVPlayer = obs.who()
#honba
honba = obs.honba()
#riichi
riichi = obs.kyotaku()

wall = len(json.loads(env.state().to_json())["hiddenState"]["wall"])



def setBoardState(roundWind, dealer, POVPlayer, honba, riichi, wall, p0Score, p1Score, p2Score, p3Score):
    """ Set the board state.
    
    """
    
    return None

class BigAttentionNet(torch.nn.Module):
    """ Attention Layer into Bigger feed-forward net. """

    def __init__(self, n_heads):
        super(BigAttentionNet, self).__init__()
        
        self.name = f"MHA-{n_heads}"
#         print(self.name)
        
        self.mha1 = torch.nn.MultiheadAttention(embed_dim=374, 
                                                num_heads=n_heads,  # 1, 11, or 34 are doable
                                                dropout=0.0,   # Default: 0.0.
                                                add_zero_attn=False,  # Default: False - Have this false, from not so many experiments, it seems like it slows down learning accuracy by a almost unoticeable bit
                                               )

        self.fc1 = torch.nn.Linear(11 * 34, 4096)   # EXTRA LAYER
        self.fc2 = torch.nn.Linear(4096, 2048)      # EXTRA LAYER
        self.fc3 = torch.nn.Linear(2048, 1024)
        self.fc4 = torch.nn.Linear(1024, 512)
        self.fc5 = torch.nn.Linear(512, 256)
        self.fc6 = torch.nn.Linear(256, 128)
        self.fc7 = torch.nn.Linear(128, 34)
        
        self.relu_1 = torch.nn.LeakyReLU()
        self.relu_2 = torch.nn.LeakyReLU()
        self.relu_3 = torch.nn.LeakyReLU()
        self.relu_4 = torch.nn.LeakyReLU()
        self.relu_5 = torch.nn.LeakyReLU()
        self.relu_6 = torch.nn.LeakyReLU()


    def forward(self, x):
        
        batch_size = x.shape[0]
        x = x.reshape(1, batch_size, 374)  #  => x.shape[0] = Batch Size
        attn_output, attn_output_weights = self.mha1(query=x, key=x, value=x, need_weights=False)  # attn_output_weights = None, if need_weights=False
        x = (x * attn_output).reshape(batch_size, 374)

        x = self.fc1(x)
        x = self.relu_1(x)
        
        x = self.fc2(x)
        x = self.relu_2(x)
        
        x = self.fc3(x)
        x = self.relu_3(x)
        
        x = self.fc4(x)
        x = self.relu_4(x)
        
        x = self.fc5(x)
        x = self.relu_5(x)
        
        x = self.fc6(x)
        x = self.relu_6(x)
        
        x = self.fc7(x)
        
        return x 


# load model
DEVICE = torch.device("cuda")
model = BigAttentionNet(1).to(DEVICE) 

# load model weights
path = "/home/lenzlaww/document/SBU/CSE537/finalProject/mjx/model_checkpoints/2025-05-02_23-25_MHA-1/epoch-015.pt"
model.load_state_dict(torch.load(path, map_location=DEVICE)['model_state'])
model.eval()

# input a board state


# Read the saved input
with open('first_input_sample.txt', 'r') as f:
    sample_str = f.read()

# Convert string to list
sample_list = ast.literal_eval(sample_str)

# Convert to tensor
sample_tensor = torch.tensor(sample_list, dtype=torch.float32)  # shape: [374]

# Expand to batch of 32
X = sample_tensor.unsqueeze(0).repeat(32, 1).to(DEVICE)  # shape: [32, 374]

print(X)  # Should be: torch.Size([32, 374])

with torch.no_grad():
    outputs = model(X)             # shape: [32, 34] (assuming 34 classes)

    # Get top-k predictions per sample
    topk_values, topk_indices = torch.topk(outputs, k=34, dim=1)

print("  Top-5 predictions:", topk_indices[0].tolist())
