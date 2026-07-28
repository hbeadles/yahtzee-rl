import gymnasium as gym
import torch
from collections import namedtuple
import torch.nn as nn
from yahtzee_rl.envs.yahtzee_env import YahtzeeEnv
from typing import Dict
import torch.optim as optim
import random
import math

transition = namedtuple("Transition", ("state", "action", "next_state", "reward", "mask", "next_mask", "upper_bonus_achieved"))

def parse_state(observation: torch.Tensor) -> Dict[str, torch.Tensor]:
    dice_values = observation[:, :5]
    roll_number = observation[:, 5:6]
    round_number = observation[:, 6:7]
    time_to_score = observation[:, 7:8]
    score_card = observation[:, 8:8+13]

    return {
        "dice_values": dice_values,
        "roll_number": roll_number,
        "round_number": round_number,
        "time_to_score": time_to_score,
        "score_card": score_card
    }

class ReplayMemory:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buffer: list = [None] * capacity
        self.write_ptr: int = 0
        self._size: int = 0

    def push(self, payload: transition):
        self._buffer[self.write_ptr] = payload
        self.write_ptr = (self.write_ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def label_episode(self, start_idx: int, n_transitions: int, achieved: bool):
        flag = 1.0 if achieved else 0.0
        for i in range(n_transitions):
            slot = (start_idx + i) % self.capacity
            t = self._buffer[slot]
            self._buffer[slot] = t._replace(upper_bonus_achieved=flag)

    def sample(self, batch_size: int):
        valid = self._buffer[:self._size]
        sample = random.sample(valid, batch_size)
        return transition(*zip(*sample))

    def __len__(self):
        return self._size



class DQN(nn.Module):

    def __init__(self, observation_state_dim: int,
                       action_dim: int,
                       device: torch.device,
                       hidden_dim: int = 128):
        super(DQN, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(observation_state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        # Q-head: same role as the old self.linear's final layer.
        # Action masking is NOT done here -- applied at selection time (see DQNAgent).
        self.q_head = nn.Linear(hidden_dim, action_dim)
        # Aux head: predicts whether the upper bonus will be achieved this episode.
        self.aux_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_head(self.backbone(x))

    def predict_bonus_achieved(self, x: torch.Tensor) -> torch.Tensor:
        return self.aux_head(self.backbone(x))
    

class DQNAgent():

    def __init__(
                self,
                observation_state_dim: int,
                action_dim: int,
                hidden_dim: int = 128,
                learning_rate: float = 1e-3,
                gamma: float = 0.99,
                epsilon_start: float = 1.0,
                epsilon_end: float = 0.01,
                epsilon_decay: float = 2500,
                target_update_freq: int = 100,
                device: torch.device = torch.device("cpu"),
                buffer_size: int = 10000,
                batch_size: int = 64,
                aux_lambda: float = 0.2,
                ):
        self.batch_size = batch_size
        self.observation_state_dim = observation_state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.replay_memory = ReplayMemory(capacity=buffer_size)
        self.network = DQN(observation_state_dim, action_dim, device, hidden_dim).to(device)
        self.target_network = DQN(observation_state_dim, action_dim, device, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon = 0.0
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.device = device
        self.criterion = nn.SmoothL1Loss()
        self.aux_lambda = aux_lambda
        self.update_target_network()

    def update_epsilon(self, steps_done: int):
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1. * steps_done / self.epsilon_decay)


    def select_action(self, state: torch.Tensor, action_mask: torch.Tensor, steps_done: int) -> int:
        # action_mask: bool tensor of shape [1, action_dim], True = legal action.
        self.update_epsilon(steps_done)

        if random.random() < self.epsilon:
            # Explore: pick uniformly among the legal actions. The mask already
            # encodes phase-specific legality (rolling vs scoring, Joker rule),
            # so we don't need to branch on time_to_score here.
            valid_actions = torch.nonzero(action_mask.squeeze(0), as_tuple=False).squeeze(-1).tolist()
            return random.choice(valid_actions)
        else:
            # Exploit: mask the OUTPUT logits, then argmax over legal actions.
            with torch.no_grad():
                state = state.to(self.device)
                q_values = self.network(state)                              # [1, action_dim]
                q_values = q_values.masked_fill(~action_mask, -float('inf'))
                return q_values.argmax(dim=1).item()
            
    def evaluate(self, state: torch.Tensor, action_mask: torch.Tensor) -> int:
        # Evaluate the policy without exploration (epsilon=0). Used for evaluation.
        with torch.no_grad():
            state = state.to(self.device)
            q_values = self.network(state)    
            q_values = q_values.masked_fill(~action_mask, -float('inf'))                          # [1, action_dim]
            return q_values.argmax(dim=1).item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def push_to_memory(self, state: transition):
        self.replay_memory.push(state)

    def update(self):
        if len(self.replay_memory) < self.batch_size:
            return

        transitions = self.replay_memory.sample(self.batch_size)
        states = torch.cat(transitions.state).to(self.device)              # [B, obs_dim]
        actions = torch.cat(transitions.action).to(self.device)           # [B]
        rewards = torch.cat(transitions.reward).to(self.device)           # [B]

        non_final_mask = torch.tensor(
            tuple(s is not None for s in transitions.next_state),
            device=self.device, dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in transitions.next_state if s is not None]
        ).to(self.device)                                                  # [Nf, obs_dim]

        # The legal-action masks for those same non-final next states, selected
        # in the same order so rows line up with non_final_next_states.
        non_final_next_masks = torch.cat(
            [m for m, s in zip(transitions.next_mask, transitions.next_state) if s is not None]
        ).to(self.device)                                                  # [Nf, action_dim]

        # Q(s_t, a_t): no masking needed -- gather pulls the action actually
        # taken, which was already legal when it was chosen.
        state_action_values = self.network(states).gather(1, actions.unsqueeze(1))

        # max_a' Q_target(s', a') over LEGAL next actions only. Masking the
        # output before the max is the part that matters for correctness:
        # otherwise we'd bootstrap off the Q-value of an illegal next action.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_q = self.target_network(non_final_next_states)            # [Nf, action_dim]
            next_q = next_q.masked_fill(~non_final_next_masks, -float('inf'))
            next_state_values[non_final_mask] = next_q.max(1)[0]

        expected_state_action_values = (next_state_values * self.gamma) + rewards
        td_loss = self.criterion(state_action_values.squeeze(), expected_state_action_values)

        # Auxiliary loss: predict whether the upper bonus was achieved this episode.
        # Transitions are labeled retroactively at episode end; None = unlabeled, skip.
        if self.aux_lambda > 0.0:
            raw_aux = list(transitions.upper_bonus_achieved)
            labeled_idx = [i for i, v in enumerate(raw_aux) if v is not None]
            if labeled_idx:
                idx_t = torch.tensor(labeled_idx, device=self.device)
                aux_targets = torch.tensor(
                    [raw_aux[i] for i in labeled_idx],
                    dtype=torch.float32, device=self.device,
                ).unsqueeze(1)
                aux_pred = self.network.predict_bonus_achieved(states[idx_t])
                aux_loss = torch.nn.functional.binary_cross_entropy(aux_pred, aux_targets)
                loss = td_loss + self.aux_lambda * aux_loss
            else:
                loss = td_loss
        else:
            loss = td_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.network.parameters(), 100)
        self.optimizer.step()
    
    def save(self, checkpoint_path):
        torch.save(self.network.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.network.load_state_dict(torch.load(checkpoint_path))
        self.target_network.load_state_dict(torch.load(checkpoint_path))
    
    def train_mode(self):
        self.network.train()
        self.target_network.train()

    def eval_mode(self):
        self.network.eval()
        self.target_network.eval()


def train(env: YahtzeeEnv, agent: DQNAgent, num_timesteps: int, tau: float = 1.0):

    for timestep in range(num_timesteps):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
        done = False
        total_reward = 0

        while not done:
                # Masks stored as [1, action_dim] so torch.cat in update() yields [B, action_dim].
            current_mask = torch.tensor(env.action_masks(), dtype=torch.bool).unsqueeze(0).to(agent.device)
            action = agent.select_action(state, current_mask)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            next_mask = torch.tensor(env.action_masks(), dtype=torch.bool).unsqueeze(0).to(agent.device)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(agent.device) if not done else None
            reward_tensor = torch.tensor([reward], dtype=torch.float32).to(agent.device)

            agent.push_to_memory(transition(state, torch.tensor([action]), next_state_tensor, reward_tensor, current_mask, next_mask, None))
            state = next_state_tensor if next_state_tensor is not None else None

            agent.update()
            target_net_state_dict = agent.target_network.state_dict()
            policy_net_state_dict = agent.network.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
            agent.target_network.load_state_dict(target_net_state_dict)

        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}")

