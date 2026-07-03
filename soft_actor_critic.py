import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
import gymnasium as gym

# ========================================== #
#               REPLAY BUFFER                #
# ========================================== #
class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        # Use np.float32 right away to avoid expensive type casting later in PyTorch
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


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

        # 📐 Add Layer Normalization
        self.ln1 = nn.LayerNorm(fc1_dims)
        self.ln2 = nn.LayerNorm(fc2_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # 📐 Apply Orthogonal Initialization
        self.init_weights()
        self.to(self.device)

    def init_weights(self):
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.q.weight, gain=1.0)

    def forward(self, state, action):
        action_value = self.fc1(torch.cat([state, action], dim=1))
        # 📐 Normalize before activation
        action_value = self.ln1(action_value)
        action_value = F.relu(action_value)
        
        action_value = self.fc2(action_value)
        # 📐 Normalize before activation
        action_value = self.ln2(action_value)
        action_value = F.relu(action_value)
        
        q = self.q(action_value)
        return q

    def save_checkpoint(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims=256, fc2_dims=256,
                 name='value', chkpt_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = nn.Linear(input_dims[0], fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.v = nn.Linear(fc2_dims, 1)

        # 📐 Add Layer Normalization
        self.ln1 = nn.LayerNorm(fc1_dims)
        self.ln2 = nn.LayerNorm(fc2_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # 📐 Apply Orthogonal Initialization
        self.init_weights()
        self.to(self.device)

    def init_weights(self):
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.v.weight, gain=1.0)

    def forward(self, state):
        state_value = self.fc1(state)
        # 📐 Normalize before activation
        state_value = self.ln1(state_value)
        state_value = F.relu(state_value)
        
        state_value = self.fc2(state_value)
        # 📐 Normalize before activation
        state_value = self.ln2(state_value)
        state_value = F.relu(state_value)
        
        v = self.v(state_value)
        return v

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

        # 📐 Add Layer Normalization
        self.ln1 = nn.LayerNorm(fc1_dims)
        self.ln2 = nn.LayerNorm(fc2_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # 📐 Apply Orthogonal Initialization
        self.init_weights()
        self.to(self.device)

    def init_weights(self):
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        # 📐 Small gain for output layers to start with a standard normal distribution
        nn.init.orthogonal_(self.mu.weight, gain=0.01)
        nn.init.orthogonal_(self.sigma.weight, gain=0.01)

    def forward(self, state):
        prob = self.fc1(state)
        # 📐 Normalize before activation
        prob = self.ln1(prob)
        prob = F.relu(prob)

        prob = self.fc2(prob)
        # 📐 Normalize before activation
        prob = self.ln2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)
        sigma = torch.clamp(sigma, min=self.reparam_noise, max=1)
        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample() # Adds noise for gradients (Exploration/Training)
        else:
            actions = probabilities.sample()  # Inference only

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
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.reward_scale = reward_scale

        # Initialize networks
        self.actor = ActorNetwork(alpha, input_dims, max_action=env.action_space.high,
                                  fc1_dims=layer1_size, fc2_dims=layer2_size, 
                                  n_actions=n_actions, name='actor')
        
        self.critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                                      fc1_dims=layer1_size, fc2_dims=layer2_size, name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                                      fc1_dims=layer1_size, fc2_dims=layer2_size, name='critic_2')
        
        self.value = ValueNetwork(beta, input_dims, fc1_dims=layer1_size, 
                                  fc2_dims=layer2_size, name='value')
        self.target_value = ValueNetwork(beta, input_dims, fc1_dims=layer1_size, 
                                         fc2_dims=layer2_size, name='target_value')

        # 🔄 Initialize Automated Temperature (Entropy) Tuning
        self.target_entropy = -n_actions
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.actor.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha)

        # Hard copy Value network parameters to Target Value network at initialization
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

        for target_param, param in zip(self.target_value.parameters(), self.value.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        reward = torch.tensor(reward, dtype=torch.float32).to(self.actor.device)
        done = torch.tensor(done).to(self.actor.device)
        state_ = torch.tensor(new_state, dtype=torch.float32).to(self.actor.device)
        state = torch.tensor(state, dtype=torch.float32).to(self.actor.device)
        action = torch.tensor(action, dtype=torch.float32).to(self.actor.device)

        # 🔄 Get current alpha value from log_alpha
        current_alpha = self.log_alpha.exp()

        # ==============================================================
        # 1. Update Value Network
        # ==============================================================
        with torch.no_grad():
            actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
            log_probs = log_probs.squeeze()
            q1_new_policy = self.critic_1(state, actions).squeeze()
            q2_new_policy = self.critic_2(state, actions).squeeze()
            critic_value = torch.min(q1_new_policy, q2_new_policy)
            # 🔄 Value target now uses dynamic alpha
            value_target = critic_value - current_alpha * log_probs

        value = self.value(state).squeeze()
        value_loss = 0.5 * F.mse_loss(value, value_target)
        
        self.value.optimizer.zero_grad()
        value_loss.backward()
        self.value.optimizer.step()

        # ==============================================================
        # 2. Update Actor Network
        # ==============================================================
        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.squeeze()
        
        q1_new_policy = self.critic_1(state, actions).squeeze()
        q2_new_policy = self.critic_2(state, actions).squeeze()
        critic_value = torch.min(q1_new_policy, q2_new_policy)

        # 🔄 Actor loss now scales entropy with dynamic alpha
        actor_loss = torch.mean(current_alpha * log_probs - critic_value)
        
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # ==============================================================
        # 3. Update Critic Networks
        # ==============================================================
        with torch.no_grad():
            value_ = self.target_value(state_).squeeze()
            value_[done] = 0.0
            q_hat = self.reward_scale * reward + self.gamma * value_
        
        q1_old_policy = self.critic_1(state, action).squeeze()
        q2_old_policy = self.critic_2(state, action).squeeze()
        
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # ==============================================================
        # 4. Update Alpha (Entropy Temperature)
        # ==============================================================
        # 🔄 Optimize alpha using the entropy loss formula
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ==============================================================
        # 5. Soft Update Target Value Network
        # ==============================================================
        self.update_network_parameters()


# ========================================== #
#               MAIN EXECUTION               #
# ========================================== #
if __name__ == '__main__':
    # ---------------------------------------------------------
    # PHASE 1: FAST TRAINING (NO GUI)
    # ---------------------------------------------------------
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
            
            # 📐 Reward Engineering: Suppress vibrations when close to the top
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

    # ---------------------------------------------------------
    # PHASE 2: TESTING & VISUALIZATION (WITH GUI)
    # ---------------------------------------------------------
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