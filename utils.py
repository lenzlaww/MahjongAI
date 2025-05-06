import numpy as np
import json

import mjx
from mjx.observation import Observation
from mjx.env import MjxEnv
from typing import Any

from discard_tile.discard import get_tiles_prob
DISCARD_MODEL_PATH = "pretrained_models/ml_discard_model.pt"

def if_i_win(obs: Observation) -> bool:
        """
        Check if the player has won.
        """
        event = obs.events()[-1]
        if event.type == mjx.EventType.RON or event.type == mjx.EventType.TSUMO:
            if event.who() == obs.who():
                return True
        return False
    
def if_i_deal_in(obs: Observation) -> bool:
    """
    Check if the player has dealt in.
    """
    cur_event = obs.events()[-1]

    if cur_event.type == mjx.EventType.RON:
        prev_event = obs.events()[-2]
        if cur_event.who() != obs.who() and prev_event.type == mjx.EventType.DISCARD:
            if prev_event.who() == obs.who():
                return True
    return False

def if_safe_tile(obs: Observation, env: MjxEnv) -> Any:
    """
    Check if the tile discarded last turn is a safe tile by using the supervised learning model: discard model.
    If the score of the tile is less than 0, it is a danger tile.
    """
    tile_id = -1
    # Get the score for all tiles
    tile_score_board = get_tiles_prob(obs, env, DISCARD_MODEL_PATH)
    # Get the last tile discarded by the target player
    for event in obs.events():
        if event.type == mjx.EventType.DISCARD or event.type == mjx.EventType.TSUMOGIRI:
            if event.who() != obs.who():
                tile_id = event.tile().id()//4
                break
    # If the tile_id is -1, it means that there is no tile discarded by the target player.
    for k, v in enumerate(tile_score_board):
        if k == tile_id:
            # If the score is less than 0, it is a danger tile.
            if v < 0:
                return 0
            else:
                return 1
    return -1
 
    
    
def get_tens_change(obs: Observation, env: MjxEnv) -> int:
    """
    Get the change in tens.
    """

    state = env.state()
    state_json = state.to_json()
    state_dict = json.loads(state_json)

    if "wins" in state_dict["roundTerminal"]:
        tens_change = state_dict["roundTerminal"]['wins'][0]['tenChanges'][obs.who()]   
        return tens_change
    return 0

def if_i_riichi(obs: Observation) -> bool:
    """
    Check if the player has declared riichi.
    """
    for event in obs.events():
        if event.type == mjx.EventType.RIICHI and event.who() == obs.who():
            return True
    return False

REWARD_WEIGHT = {
    'shanten_change': 5,  # shanten change
    'tenpei_bouns': 10, # tenpei bouns
    'win_base': 20, # win base
    'deal_in_penalty': -30, # deal in penalty
    'safe_tile_bounus': 1, # safe tile bounus
    'danger_tile_penalty': -2, # danger tile penalty
    'riichi_success': 8, # riichi success
    'riichi_fail': -4, # riichi fail
    'draw_tenpei': 5, # draw tenpei
    'draw_notenpei': -5, # draw no tenpei
}

def compute_reward(cur_obs: Observation, prev_obs: Observation, env: MjxEnv, discard_model: bool = True) -> float:
    """
    Compute the reward based on the observation.
    """
    reward = 0

    # Reward when Round is Continuing
    # Get Shanten Number
    cur_shanten = cur_obs.curr_hand().shanten_number()
    prev_shanten = prev_obs.curr_hand().shanten_number()

    shanten_change = prev_shanten - cur_shanten
    reward += shanten_change * REWARD_WEIGHT['shanten_change']
    
    
    if cur_shanten == 1:
        reward += REWARD_WEIGHT['tenpei_bouns']

    # Reward safe tile
    if discard_model:
        safe = if_safe_tile(cur_obs, env)
        if safe == 0:
            reward += REWARD_WEIGHT['danger_tile_penalty']
        elif safe == 1:
            reward += REWARD_WEIGHT['safe_tile_bounus']
    # Reward when round is done
    riichi_flag = if_i_riichi(cur_obs)
    # Round bonus
    if env.done('round'):
        draw_ended = False
        if cur_obs.events()[-1].type == mjx.EventType.DRAW:
            draw_ended = True
        if draw_ended:
            if cur_obs.curr_hand().shanten_number() == 1:
                reward += REWARD_WEIGHT['draw_tenpei']
            else:
                reward += REWARD_WEIGHT['draw_notenpei']
        else:
            # win bonus
            if if_i_win(cur_obs):
                reward += REWARD_WEIGHT['win_base']
                if riichi_flag:
                    reward += REWARD_WEIGHT['riichi_success']

            else:
                if riichi_flag:
                    reward += REWARD_WEIGHT['riichi_fail']
            # Penalty for dealing in
            if if_i_deal_in(cur_obs):
                reward += REWARD_WEIGHT['deal_in_penalty']
        reward += get_tens_change(cur_obs, env)//100
        try:
            reward += env.rewards()['player_0']//100
        except:
            pass
    return reward
        
    
