import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
import gymnasium as gym

# ========================================== #
#          PRIORITIZED REPLAY BUFFER         #
# ========================================== #
class PrioritizedReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.alpha = alpha  # 🔄 Controls how much prioritization is used (0 = uniform, 1 = full prioritization)
        self.beta = beta    # 🔄 Controls importance sampling correction
        self.beta_increment = beta_increment
        
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        
        # 📊 Array to store priorities
        self.priorities = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        
        # 📐 New transitions get max priority to ensure they are sampled at least once
        max_priority = np.max(self.priorities) if self.mem_cntr > 0 else 1.0
        self.priorities[index] = max_priority
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        
        # 🧮 Calculate sampling probabilities based on priorities
        priorities = self.priorities[:max_mem]
        probabilities = (priorities ** self.alpha) / np.sum(priorities ** self.alpha)
        
        # 🎲 Choice based on calculated probabilities
        batch = np.random.choice(max_mem, batch_size, p=probabilities, replace=False)

        # ⚖️ Importance Sampling weights computation to fix bias
        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = (max_mem * probabilities[batch]) ** (-self.beta)
        weights /= np.max(weights)  # Normalize weights
        weights = np.array(weights, dtype=np.float32)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones, batch, weights

    def update_priorities(self, batch_indices, td_errors):
        # 🔄 Update priorities in buffer with new TD-errors from the learning step
        for idx, error in zip(batch_indices, td_errors):
            self.priorities[idx] = np.abs(error) + 1e-6  # small constant to avoid zero priority


# ========================================== #
#               NEURAL NETWORKS              #
# ========================================== #
class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=256,
                 name='critic', chkpt_dir='tmp/sac'):
        super(CriticNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = nn.Linear(input_dims[0] + n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        action_value = self.fc1(torch.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)
        q = self.q(action_value)
        return q

    def save_checkpoint(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, fc1_dims=256,
                 fc2_dims=256, n_actions=2, name='actor', chkpt_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.max_action = max_action
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(input_dims[0], fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mu = nn.Linear(fc2_dims, n_actions)
        self.sigma = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)
        mu = self.mu(prob)
        sigma = self.sigma(prob)
        sigma = torch.clamp(sigma, min=self.reparam_noise, max=1)
        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action_tanh = torch.tanh(actions)
        action = action_tanh * torch.tensor(self.max_action).to(self.device)
        
        log_probs = probabilities.log_prob(actions)
        log_probs -= torch.log(1 - action_tanh.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)
        return action, log_probs

    def save_checkpoint(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


# ========================================== #
#                    AGENT                   #
# ========================================== #
class Agent:
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=[8],
                 env=None, gamma=0.99, n_actions=2, max_size=1000000, tau=0.005,
                 layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2):
        self.gamma = gamma
        self.tau = tau
        # 🔄 Initialize Prioritized Buffer
        self.memory = PrioritizedReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.reward_scale = reward_scale

        self.actor = ActorNetwork(alpha, input_dims, max_action=env.action_space.high,
                                  fc1_dims=layer1_size, fc2_dims=layer2_size, 
                                  n_actions=n_actions, name='actor')
        
        self.critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                                      fc1_dims=layer1_size, fc2_dims=layer2_size, name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                                      fc1_dims=layer1_size, fc2_dims=layer2_size, name='critic_2')
        
        self.target_critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                                             fc1_dims=layer1_size, fc2_dims=layer2_size, name='target_critic_1')
        self.target_critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                                             fc1_dims=layer1_size, fc2_dims=layer2_size, name='target_critic_2')

        self.target_entropy = -n_actions
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.actor.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha)

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        self.actor.eval()
        with torch.no_grad():
            state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.actor.device)
            actions, _ = self.actor.sample_normal(state, reparameterize=False)
        self.actor.train()
        return actions.cpu().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        # 🔄 Receive index and sampling weights from the buffer
        state, action, reward, new_state, done, idxs, weights = self.memory.sample_buffer(self.batch_size)

        reward = torch.tensor(reward, dtype=torch.float32).to(self.actor.device)
        done = torch.tensor(done).to(self.actor.device)
        state_ = torch.tensor(new_state, dtype=torch.float32).to(self.actor.device)
        state = torch.tensor(state, dtype=torch.float32).to(self.actor.device)
        action = torch.tensor(action, dtype=torch.float32).to(self.actor.device)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.actor.device)

        current_alpha = self.log_alpha.exp()

        # ==============================================================
        # 1. Update Critic Networks
        # ==============================================================
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample_normal(state_, reparameterize=True)
            next_log_probs = next_log_probs.squeeze()
            
            target_q1 = self.target_critic_1(state_, next_actions).squeeze()
            target_q2 = self.target_critic_2(state_, next_actions).squeeze()
            min_target_q = torch.min(target_q1, target_q2) - current_alpha * next_log_probs
            
            q_hat = self.reward_scale * reward + (1.0 - done.float()) * self.gamma * min_target_q
        
        q1_old_policy = self.critic_1(state, action).squeeze()
        q2_old_policy = self.critic_2(state, action).squeeze()
        
        # ⚖️ Critic loss multiplied by Importance Sampling weights
        critic_1_errors = F.mse_loss(q1_old_policy, q_hat, reduction='none')
        critic_2_errors = F.mse_loss(q2_old_policy, q_hat, reduction='none')
        
        critic_1_loss = torch.mean(weights * critic_1_errors)
        critic_2_loss = torch.mean(weights * critic_2_errors)
        critic_loss = 0.5 * (critic_1_loss + critic_2_loss)
        
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # 🔄 Update transition priorities based on new TD-errors
        with torch.no_grad():
            td_errors = (torch.abs(q1_old_policy - q_hat) + torch.abs(q2_old_policy - q_hat)) / 2.0
            self.memory.update_priorities(idxs, td_errors.cpu().numpy())

        # ==============================================================
        # 2. Update Actor Network
        # ==============================================================
        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.squeeze()
        
        q1_new_policy = self.critic_1(state, actions).squeeze()
        q2_new_policy = self.critic_2(state, actions).squeeze()
        critic_value = torch.min(q1_new_policy, q2_new_policy)

        actor_loss = torch.mean(current_alpha * log_probs - critic_value)
        
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # ==============================================================
        # 3. Update Alpha (Entropy Temperature)
        # ==============================================================
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ==============================================================
        # 4. Soft Update Target Critic Networks
        # ==============================================================
        self.update_network_parameters()


# ========================================== #
#               MAIN EXECUTION               #
# ========================================== #
if __name__ == '__main__':
    env = gym.make('Pendulum-v1')
    
    input_dims = env.observation_space.shape
    n_actions = env.action_space.shape[0]

    agent = Agent(input_dims=input_dims, env=env, n_actions=n_actions)
    
    n_games = 250
    score_history = []

    print("Phase 1: Starting fast training (No GUI) - Please wait...")

    for i in range(n_games):
        observation, info = env.reset()
        done = False
        truncated = False
        score = 0
        
        while not (done or truncated):
            action = agent.choose_action(observation)
            observation_, reward, done, truncated, info = env.step(action)
            
            if observation[0] > 0.95:
                penalty = 0.5 * (observation[2] ** 2)
                reward -= penalty

            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            
            score += reward
            observation = observation_
            
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        
        print(f"Episode {i:03d} | Score: {score:7.1f} | 100-Game Avg: {avg_score:7.1f}")

    print("Training finished successfully!")
    env.close()

    print("\nPhase 2: Let's see how the trained agent performs!")
    test_env = gym.make('Pendulum-v1', render_mode='human')
    n_test_games = 5
    
    for i in range(n_test_games):
        observation, info = test_env.reset()
        done = False
        truncated = False
        score = 0
        
        while not (done or truncated):
            action = agent.choose_action(observation)
            observation_, reward, done, truncated, info = test_env.step(action)
            
            score += reward
            observation = observation_
            
        print(f"Test Episode {i+1}/{n_test_games} | Final Score: {score:7.1f}")

    test_env.close()