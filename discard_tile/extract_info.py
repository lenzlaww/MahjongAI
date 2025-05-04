# get the x input for the model
# x input shape = tensor[32,374]

import torch
import mjx
import json
from mjx.open import Open


def decode_open(open_list):
    open_all = []
    for open in open_list:
        open_obj = Open(open)
        for tile in open_obj.tiles():
            open_all.append(tile.id()//4)
    return open_all

def findwall(obs: mjx.Observation) -> List[int]:
    draw_cnt = 0
    for event in obs.events():
        # print(event.type())
        if event.type() == mjx.EventType.DRAW:
            draw_cnt += 1
    print(f"remain wall: {69-draw_cnt + 1}")
    return 69-draw_cnt + 1

def map_score(score, x_min=0, x_max=50000, y_min=-128, y_max=127):
    return int(round(y_min + (score - x_min) * (y_max - y_min) / (x_max - x_min)))

def process_obs(obs_dict: dict, env) -> torch.Tensor:
    """
    extract info from obs_dict, and convert to a PyTorch tensor of shape [32, 374].

    Parameters:
    - obs_dict: a dictionary of observations (dict)
    - env: the mjx.env.MjxEnv environment object

    Returns:
    - tensor: PyTorch tensor of shape [32, 374]
    """

    obs = obs_dict['player_0']
    state = env.state()
    state_json = state.to_json()
    state_dict = json.loads(state_json)


    roundWind = obs.round() # FIX THIS!!!!
    dealer = obs.dealer()
    POVPlayer = 0
    #honba
    honba = obs.honba()
    #riichi
    riichi = obs.kyotaku()

    wall = findwall(obs)

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
        return None

    p0_score = map_score(p0_score)
    p1_score = map_score(p1_score)
    p2_score = map_score(p2_score)
    p3_score = map_score(p3_score)

    dora = state_dict["publicObservation"]["doraIndicators"][0] // 4

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
        return None

    # melds
    player_ids = state_dict["publicObservation"]["playerIds"]
    melds_info ={}
    for index, id in enumerate(player_ids):

        try:
            open_list = state_dict["privateObservations"][index]["currHand"]["opens"]
            melds_info[id] = decode_open(open_list)
        except:
            melds_info[id] = []

    

    return None