import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-5         # learning rate of the actor 
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0.000   # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, **kwargs):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """

        self.state_size = kwargs['state_size']
        self.action_size = kwargs['action_size']
        self.seed = kwargs['random_seed']
        self.num_agents = kwargs['num_agents']
        self.iter = 0
        self.noise_scale = 1.0
        
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(2*self.state_size, 2*self.action_size, self.seed).to(device)
        self.actor_target = Actor(2*self.state_size, 2*self.action_size, self.seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=200, gamma=0.99)
        #self.actor_optimizer = optim.Adam(self.actor_local.parameters())
        
        # Critic Network (w/ Target Network)
        self.critic_local = Critic(2*self.state_size, 2*self.action_size, self.seed).to(device)
        self.critic_target = Critic(2*self.state_size, 2*self.action_size, self.seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=200, gamma=0.99)
        #self.critic_optimizer = optim.Adam(self.critic_local.parameters())
        
        self.hard_copy_weights(self.actor_target, self.actor_local)
        self.hard_copy_weights(self.critic_target, self.critic_local)

        # Noise process
        self.noise = OUNoise(self.action_size, self.seed)

        # Replay memory
        self.memory = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE, self.seed)
        # Prioritized replay memory
        #self.memory = NaivePrioritizedBuffer(BUFFER_SIZE, BATCH_SIZE, self.seed)

        if 'actor_chkpt_file' in kwargs and 'critic_chkpt_file' in kwargs:
            checkpoint_actor = torch.load(kwargs['actor_chkpt_file'])
            checkpoint_critic = torch.load(kwargs['critic_chkpt_file'])
            self.actor_local.load_state_dict(checkpoint_actor)
            self.critic_local.load_state_dict(checkpoint_critic)
            checkpoint_actor_t = torch.load(kwargs['actor_chkpt_file_t'])
            checkpoint_critic_t = torch.load(kwargs['critic_chkpt_file_t'])
            self.actor_target.load_state_dict(checkpoint_actor_t)
            self.critic_target.load_state_dict(checkpoint_critic_t)

    def hard_copy_weights(self, target, source):
        """ copy weights from source to target network (part of initialization)"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        state_c = state.reshape((1, 2*self.state_size))
        next_state_c = next_state.reshape((1, 2*self.state_size))
        action_c = action.reshape((1, 2*self.action_size))
        reward_c = np.mean(reward)
        done_c = np.any(done)
        
        #print(next_state_c)
        self.memory.add(state_c, action_c, reward_c, next_state_c, done_c)
        #self.memory.add(state_c, action[1], reward[1], next_state[1], next_state_c, done[1])
        
        # Learn, if enough samples are available in memory
        self.iter = self.iter+1
        self.iter = self.iter%1
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

        #self.actor_scheduler.step()
        #self.critic_scheduler.step()
        
    def act(self, state, add_noise=True, noise_scale=1.0):
        """Returns actions for given state as per current policy."""

        action = np.zeros((self.num_agents, self.action_size))
        s = state.reshape(1, 48)
        
        s1 = torch.from_numpy(s).float().to(device) 
        #s2 = torch.from_numpy(np.expand_dims(state[1], axis=0)).float().to(device)
        
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(s1).cpu().data.numpy().squeeze(axis=0)
            action[0] = actions[0:2]
            action[1] = actions[2:4]
            #action[1] = self.actor_local(s2).cpu().data.numpy().squeeze(axis=0)
            
        self.actor_local.train()
        if add_noise:
            action[0] += self.noise.sample()*noise_scale
            action[1] += self.noise.sample()*noise_scale

            
            
        np.clip(action, -1, 1)
        return action

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        #states, actions, rewards, next_states, dones, indices, weights = experiences
        ( states, actions, rewards, next_states, dones ) = experiences
        
        #print(experiences)
        #print(states)
        #print(actions)
        #print(next_states)

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        #print(actions_next)
        Q_targets_next = self.critic_target(next_states, actions_next)
        #print(Q_targets_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        #print(Q_targets)
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        #print(Q_expected)
        
        critic_loss = F.mse_loss(Q_expected, Q_targets)     
        
        #weights_tensor = torch.tensor(weights)
        #print(weights)
        #print(weights_tensor)
        
        #critic_loss  = (Q_targets - Q_expected).pow(2) * weights_tensor
        #critic_loss  = (Q_targets - Q_expected).pow(2)
        #prios = critic_loss + 1e-5
        #critic_loss  = critic_loss.mean()
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        #self.memory.update_priorities(indices, prios.data.cpu().numpy())
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        #self.sigma = 0.99*self.sigma
        #self.theta = 0.99*self.theta
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
class NaivePrioritizedBuffer(object):
    def __init__(self, capacity, batch_size, seed, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity   = capacity
        self.memory     = []
        self.pos        = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        np.random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        assert state.ndim == next_state.ndim
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        
        max_prio = self.priorities.max() if self.memory else 1.0
        
        if len(self.memory) < self.capacity:
            self.memory.append((state, action, reward, next_state, done))
        else:
            self.memory[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, beta=0.4):
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs  = prios ** self.prob_alpha
        probs /= probs.sum()
        
        #indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
        indices = np.random.choice(len(self.memory), self.batch_size, False, p=probs)
        #indices = np.random.randint(0, len(self.memory), self.batch_size)
        
        states = torch.from_numpy(np.vstack([self.memory[idx][0] for idx in indices if indices is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([self.memory[idx][1] for idx in indices if indices is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([self.memory[idx][2] for idx in indices if indices is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([self.memory[idx][3] for idx in indices if indices is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([self.memory[idx][4] for idx in indices if indices is not None]).astype(np.uint8)).float().to(device)
    
        total    = len(self.memory)
        weights  = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights  = np.array(weights, dtype=np.float32)
        weights = np.vstack(weights)
        
        
        return (states, actions, rewards, next_states, dones, indices, weights)
        #return (states, actions, rewards, next_states, dones)
    
    def update_priorities(self, batch_indices, batch_priorities):
        
        #print(batch_indices)
        #print(batch_priorities)
        
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.memory)        