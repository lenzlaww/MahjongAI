# get the x input for the model
# x input shape = tensor[32,374]

import torch
import mjx
import json
from mjx.open import Open
from typing import List
import pandas as pd
# torch.set_printoptions(threshold=float('inf'))


def decode_open(open_list):
    open_all = []
    for open in open_list:
        open_obj = Open(open)
        for tile in open_obj.tiles():
            open_all.append(tile.id()//4)
    return open_all

def findwall(obs: mjx.Observation) -> int:
    draw_cnt = 0
    for event in obs.events():
        # print(event.type())
        if event.type() == mjx.EventType.DRAW:
            draw_cnt += 1
    print(f"remain wall: {69 - draw_cnt + 1}")
    return 69 - draw_cnt + 1

def map_score(score, x_min=0, x_max=50000, y_min=-128, y_max=127):
    return int(round(y_min + (score - x_min) * (y_max - y_min) / (x_max - x_min)))

def fill_tensor(tensor: torch.Tensor, index_start: int, index_end: int, value: list) -> torch.Tensor:
    for v in value:
        if index_start + v < index_end:
            tensor[:, index_start+v] += 1
        else:
            raise ValueError(f"Index {index_start+v} out of bounds for tensor of size {tensor.size()}")
    return tensor

def process_obs(obs_dict: dict, env) -> torch.Tensor:

    """
    extract info from obs_dict, and convert to a PyTorch tensor of shape [32, 374].

    Parameters:
    - obs_dict: a dictionary of observations (dict)
    - env: the mjx.env.MjxEnv environment object

    Returns:
    - tensor: PyTorch tensor of shape [32, 374]
    """

    # Initialize a tensor of shape [32, 374]
    tensor = torch.zeros((32, 374), dtype=torch.int64)

    obs = obs_dict['player_0']
    state = env.state()
    state_json = state.to_json()
    state_dict = json.loads(state_json)


    roundWind = obs.round() # FIX THIS!!!!
    tensor[0, 0] = torch.tensor(roundWind, dtype=torch.int64)

    dealer = obs.dealer()
    tensor[0, 1] = torch.tensor(dealer, dtype=torch.int64)

    POVPlayer = 0
    tensor[0, 2] = torch.tensor(POVPlayer, dtype=torch.int64)
    #honba
    honba = obs.honba()
    tensor[0, 3] = torch.tensor(honba, dtype=torch.int64)

    #riichi
    riichi = obs.kyotaku()
    tensor[0, 4] = torch.tensor(riichi, dtype=torch.int64)

    wall = findwall(obs)
    tensor[0, 5] = torch.tensor(wall, dtype=torch.int64)

    p0_score = 0
    p1_score = 0
    p2_score = 0
    p3_score = 0

    try:
        for pid, obs in obs_dict.items():
            if pid == "player_0":
                p0_score = state_dict["roundTerminal"]["finalScore"]["tens"][obs.who()]
            elif pid == "player_1":
                p1_score = state_dict["roundTerminal"]["finalScore"]["tens"][obs.who()]
            elif pid == "player_2":
                p2_score = state_dict["roundTerminal"]["finalScore"]["tens"][obs.who()]
            elif pid == "player_3":
                p3_score = state_dict["roundTerminal"]["finalScore"]["tens"][obs.who()]

            score = state_dict["roundTerminal"]["finalScore"]["tens"][obs.who()]
            print(f"{pid} score: {score}")
    except KeyError:
        print("player score extract error")

    p0_score = map_score(p0_score)
    tensor[0, 6] = torch.tensor(p0_score, dtype=torch.int64)
    p1_score = map_score(p1_score)
    tensor[0, 7] = torch.tensor(p1_score, dtype=torch.int64)
    p2_score = map_score(p2_score)
    tensor[0, 8] = torch.tensor(p2_score, dtype=torch.int64)
    p3_score = map_score(p3_score)
    tensor[0, 9] = torch.tensor(p3_score, dtype=torch.int64)

    tensor[:, 10:14] = 0
    tensor[:, 14:34] = -128

    dora = state_dict["publicObservation"]["doraIndicators"]
    dora = [x//4 for x in dora]

    # POVhand
    try:
        for pid, obs in obs_dict.items():
            # print(f"{pid} hand tile: {obs.curr_hand()}")
            if pid == "player_0":
                break

        hand = obs.curr_hand().to_json()
        hand_dict = json.loads(hand)

        handlist = []
        for key, values in hand_dict.items():
            print(f"{key} hand tile: {values}")
            if key == "closedTiles":
                for value in values:
                    handlist.append(value//4)
            elif key == "opens":
                open_all = decode_open(values)
                handlist.extend(open_all)
    except KeyError:
        print("POV hand extract error")


    # fill tensor with hand tiles
    tensor = fill_tensor(tensor, 68, 102, handlist)

    # melds
    player_ids = state_dict["publicObservation"]["playerIds"]
    melds_info ={}
    for index, id in enumerate(player_ids):

        try:
            open_list = state_dict["privateObservations"][index]["currHand"]["opens"]
            melds_info[id] = decode_open(open_list)
        except:
            melds_info[id] = []
    
    for key in melds_info:
        melds = melds_info[key]
        if key == "player_0":
            tensor = fill_tensor(tensor, 102, 136, melds)
        elif key == "player_1":
            tensor = fill_tensor(tensor, 136, 170, melds)
        elif key == "player_2":
            tensor = fill_tensor(tensor, 170, 204, melds)
        elif key == "player_3":
            tensor = fill_tensor(tensor, 204, 238, melds)

    
    # pools
    pools_info ={}
    for index, id in enumerate(player_ids):
        try:
            open_list = state_dict["privateObservations"][index]['drawHistory']
            pools_info[id] = open_list
        except:
            pools_info[id] = []

    for key in pools_info:
        pools = pools_info[key]
        if key == "player_0":
            tensor = fill_tensor(tensor, 238, 272, pools)
        elif key == "player_1":
            tensor = fill_tensor(tensor, 272, 306, pools)
        elif key == "player_2":
            tensor = fill_tensor(tensor, 306, 340, pools)
        elif key == "player_3":
            tensor = fill_tensor(tensor, 340, 374, pools)

    return tensor

if __name__ == "__main__":

    # create a new tensor 
    tensor = torch.zeros((32, 374), dtype=torch.int64)
    print(tensor)
    tensor[0, 0] = torch.tensor(111, dtype=torch.int64)
    # tensor = fill_tensor(tensor, 340, 374, [0, 0, 1, 33, 33])
    print(tensor)

    # print tensor shape
    print(tensor.shape)