import gym
import numpy as np
import mjx
from mjx.agents import RandomAgent, ShantenAgent  # 使用 Shanten agent
from utils import compute_reward
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import matplotlib.pyplot as plt
import json



class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ActorCritic, self).__init__()
        
        self.base = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.actor = nn.Linear(hidden_dim, output_dim)
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        base_output = self.base(x)
        action_logits = self.actor(base_output)
        state_values = self.critic(base_output)
        return action_logits, state_values

class PPOAgent:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.01, gamma=0.99, clip_ratio=0.4, value_coef=0.5, entropy_coef=0.01, pretrained_model = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
        self.model = ActorCritic(input_dim, hidden_dim, output_dim).to(self.device)

        # If there is a pretrained model, load it
        if pretrained_model is not None:
            state_dict = torch.load(pretrained_model)
            state_dict['base.0.weight'] = state_dict['base.0.weight'][:, :544]
            
            self.model.load_state_dict(state_dict)
            print(f"Loaded pretrained model from {pretrained_model}")
            
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Collect episode data
        self.states = []
        self.action_masks = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
    
    def act(self, state, action_mask):
        state = torch.FloatTensor(state).flatten().to(self.device)
        mask = torch.FloatTensor(action_mask).to(self.device)
        
        with torch.no_grad():
            action_logits, state_value = self.model(state)
            action_logits = action_logits - (1 - mask) * 1e9
            
            dist = Categorical(logits=action_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        self.states.append(state)
        self.action_masks.append(mask)
        self.actions.append(action)
        self.values.append(state_value)
        self.log_probs.append(log_prob)
        
        return action.item()
    
    def store_reward(self, reward):
        self.rewards.append(reward)
    
    def update(self, next_state=None, done=True):
        # Compute returns and advantages
        if not done and next_state is not None:
            next_state = torch.FloatTensor(next_state).flatten().to(self.device)
            with torch.no_grad():
                _, next_value = self.model(next_state)
            last_value = next_value.item()
        else:
            last_value = 0
        
        states = torch.stack(self.states)
        action_masks = torch.stack(self.action_masks)
        actions = torch.stack(self.actions)
        old_log_probs = torch.stack(self.log_probs)
        old_values = torch.cat(self.values)
        
        returns = []
        advantages = []
        R = last_value
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = returns - old_values.detach()
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        action_logits, state_values = self.model(states)
        
        for i in range(len(action_logits)):
            action_logits[i] = action_logits[i] - (1 - action_masks[i]) * 1e9
        
        dist = Categorical(logits=action_logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        ratios = torch.exp(new_log_probs - old_log_probs.detach())
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        value_loss = nn.MSELoss()(state_values.squeeze(), returns)
        loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        
        # Reset episode data
        self.states = []
        self.action_masks = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        
        return {
            'actor_loss': actor_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': loss.item()
        }


import gym
import numpy as np
import mjx
from mjx.agents import RandomAgent, ShantenAgent  # 使用 Shanten agent
from utils import compute_reward
from collections import OrderedDict

class GymEnv(gym.Env):
    def __init__(
        self, 
        opponent_agents: list, 
        reward_type: str = "game_tenhou_7dan", 
        done_type: str = "game", 
        feature_type: str = "mjx-small-v0",
        info_type: str = "perfect",
        discard_model: bool = True,
    ):
        super(GymEnv, self).__init__()
        self.opponent_agents = opponent_agents  # Opponent agents
        self.reward_type = reward_type
        self.done_type = done_type
        self.feature_type = feature_type

        self.target_player = "player_0"  # Target player
        self.mjx_env = mjx.MjxEnv()
        self.curr_obs_dict = self.mjx_env.reset()

        self.prev_obs = None
        self.info_type = info_type

        self.discard_model = discard_model
        
        obs = next(iter(self.curr_obs_dict.values()))
        sample_feat = obs.to_features(self.feature_type)

        self.full_info = OrderedDict(
            (f"player_{i}", np.zeros_like(sample_feat)) for i in range(4)
        )

    def _update_full_info(self, obs_dict):
        # print(obs_dict)
        for player_id, obs in obs_dict.items():
            self.full_info[player_id] = obs.to_features(self.feature_type)
    
    def _set_opponent_agents(self, agents):
        """Set opponent agents"""
        self.opponent_agents = agents

    def _set_info_type(self, info_type):
        """Set information type"""
        self.info_type = info_type

    def reset(self):
        """Reset the environment and return the initial observation"""
        self.curr_obs_dict = self.mjx_env.reset()
        obs = next(iter(self.curr_obs_dict.values()))
        sample_feat = obs.to_features(self.feature_type)
        self.full_info = {
            f"player_{i}": np.zeros_like(sample_feat) for i in range(4)
        }
        self.prev_obs = None

        # Skip to the target player's turn
        while self.target_player not in self.curr_obs_dict:
            action_dict = {}
            for i, (player_id, obs) in enumerate(self.curr_obs_dict.items()):
                agent = self.opponent_agents[i]
                if type(agent) == PPOAgent:
                    feat = obs.to_features(self.feature_type).flatten()
                    legal_actions = obs.legal_actions()
                    action_index = int(agent.act(feat, obs.action_mask()))
                    action = mjx.Action.select_from(action_index, legal_actions)

                else:
                    action = agent.act(obs)
                action_dict[player_id] = action

            self.curr_obs_dict = self.mjx_env.step(action_dict)
        self._update_full_info(self.curr_obs_dict)

        # Return the observation of the target player
        obs = self.curr_obs_dict[self.target_player]
        if self.info_type == "perfect":
            feat = np.concatenate(list(self.full_info.values()), axis=-1)
        else:
            feat = obs.to_features(self.feature_type)
        mask = obs.action_mask()  # Get valid actions
        return feat, {"action_mask": mask}

    def step(self, action):
        """Execute a step and return the next state, reward, and done status"""
        action_dict = {self.target_player: mjx.Action.select_from(action, self.curr_obs_dict[self.target_player].legal_actions())}

        reward = 0

        # Iterate through the opponent agents
        for i, (pid, obs) in enumerate(self.curr_obs_dict.items()):
            if pid != self.target_player:
                if type(self.opponent_agents[i-1]) != PPOAgent:
                    action = self.opponent_agents[i-1].act(obs)
                    action_dict[pid] = action
                else:
                    # Use PPO agent to select action
                    feat = obs.to_features(self.feature_type).flatten()
                    legal_actions = obs.legal_actions()
                    action_index = int(self.opponent_agents[i-1].act(feat, obs.action_mask()))
                    action = mjx.Action.select_from(action_index, legal_actions)
                    action_dict[pid] = action


        # Update the environment with the action
        self.curr_obs_dict = self.mjx_env.step(action_dict)
        self._update_full_info(self.curr_obs_dict)
        

        # Check if it's the target player's turn
        while self.target_player not in self.curr_obs_dict:
            action_dict = {}
            for i, (player_id, obs) in enumerate(self.curr_obs_dict.items()):
                agent = self.opponent_agents[i]
                if type(agent) == PPOAgent:
                    feat = obs.to_features(self.feature_type).flatten()
                    legal_actions = obs.legal_actions()
                    action_index = int(agent.act(feat, obs.action_mask()))
                    action = mjx.Action.select_from(action_index, legal_actions)

                else:
                    action = agent.act(obs)
                action_dict[player_id] = action
            self.curr_obs_dict = self.mjx_env.step(action_dict)
            self._update_full_info(self.curr_obs_dict)

            if self.mjx_env.done(self.done_type):
                obs = list(self.curr_obs_dict.values())[0]  # Obtain the last observation
                feat = obs.to_features(self.feature_type)
                done = True
                if self.prev_obs is not None and obs is not None:
                    reward = compute_reward(self.prev_obs, obs, self.mjx_env, discard_model=self.discard_model)
                if self.target_player in self.curr_obs_dict:
                    self.prev_obs = self.curr_obs_dict[self.target_player]
                return feat, reward, done, {"action_mask": np.ones(181)}  # set action mask to all ones

        # When the game is not done, return the observation of the target player
        assert self.target_player in self.curr_obs_dict
        obs = self.curr_obs_dict[self.target_player]
        done = self.mjx_env.done(self.done_type)
        if self.prev_obs is not None and obs is not None:
            reward = compute_reward(self.prev_obs, obs, self.mjx_env, discard_model=self.discard_model)
        if self.target_player in self.curr_obs_dict:
            self.prev_obs = self.curr_obs_dict[self.target_player]
        # feat = obs.to_features(self.feature_type)
        if self.info_type == "perfect":
            # print full info
            if list(self.full_info.keys())[0] != 'player_0':
                print(f"Warning: {self.full_info.keys()} is not the same as ['player_0', 'player_1', 'player_2', 'player_3']")

            feat = np.concatenate(list(self.full_info.values()), axis=-1)
            

        else:
            feat = obs.to_features(self.feature_type)
        mask = obs.action_mask()
        
        return feat, reward, done, {"action_mask": mask}


def plot_rewards(rewards, path="logs/reward_curve.png"):
    plt.figure()
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward Curve")
    plt.savefig(path)
    plt.close()

def train_curriculum_agent(
    info_type = "default", opponents = list(RandomAgent() for _ in range(3)),
    num_episodes=1000, log_interval = 100,
    pretrained_model=None,
    stage = 1,
    all_logs = {
        "all_rewards": [],
        "all_actor_loss": [],
        "all_value_loss": [],
    },
    
    patience=100,
    delta=20,
):
    env = GymEnv(opponent_agents=opponents, info_type=info_type)
    obs, info = env.reset()
    obs_shape = obs.flatten().shape[0]
    action_dim = len(info["action_mask"])
    agent = PPOAgent(
        input_dim=obs_shape,
        hidden_dim=128,
        output_dim=action_dim,
        pretrained_model=pretrained_model, # If have a pretrained model, load it
    )

    rolling_rewards = []
    best_reward = -np.inf

    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(obs, info["action_mask"])
            next_obs, reward, done, info = env.step(action)
            agent.store_reward(reward)
            obs = next_obs
            total_reward += reward

        stats = agent.update(next_obs, done)
        all_logs["all_actor_loss"].append(stats["actor_loss"])
        all_logs["all_value_loss"].append(stats["value_loss"])
        all_logs["all_rewards"].append(total_reward)

        if episode % log_interval == 0:
            avg_reward = np.mean(all_logs["all_rewards"][-log_interval:])
            rolling_rewards.append(avg_reward)
            print(f"Episode {episode}/{num_episodes}, avg reward: {avg_reward:.3f}")
            # Update the best model if the average reward is better than the best reward
            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save(agent.model.state_dict(), f"logs/ppo_cr_cl/best_model_ppo5_stage_{stage}.pt")
                print(f"Best model saved with reward: {best_reward:.3f}")
            
            log_data = {
                "episode": episode,
                "avg_reward": avg_reward,
                "actor_loss": stats["actor_loss"],
                "value_loss": stats["value_loss"],
                "entropy": stats["entropy"],
                "total_loss": stats["total_loss"],
            }

            with open("logs/ppo_cr_cl/stage_{stage}_logs.json", "a") as f:
                f.write(json.dumps(log_data) + "\n")
            print(f"Episode {episode} logs saved.")

    
    plot_rewards(rolling_rewards, path=f"logs/ppo_cr_cl/stage_{stage}_reward_curve.png")
    return agent, all_logs