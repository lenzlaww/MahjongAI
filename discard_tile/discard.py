import torch
import ast 
import mjx
import random
from typing import List
from tqdm import tqdm
from mjx.agents import RandomAgent
from mjx.observation import Observation
import json
from discard_tile.extract_info import process_obs
# from extract_info import process_obs




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



def get_tiles_prob(obs: Observation, env, path) -> zip:
    DEVICE = torch.device("cuda")
    model = BigAttentionNet(1).to(DEVICE) 
    model.load_state_dict(torch.load(path, map_location=DEVICE)['model_state'])
    model.eval()

    X = process_obs(obs, env)


    with torch.no_grad():
        X = X.float()
        outputs = model(X)   
                  # shape: [32, 34] (assuming 34 classes)

        # Get top-k predictions per sample
        topk_values, topk_indices = torch.topk(outputs, k=34, dim=1)
        topk_values = torch.round(topk_values * 10000) / 10000
    return zip(topk_indices[0].tolist(), topk_values[0].tolist())

    