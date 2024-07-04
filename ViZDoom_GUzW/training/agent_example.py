from typing import List, Tuple
import random
import vizdoom as vzd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import skimage.transform


learning_rate = 0.00025
discount_factor = 0.99
train_epochs = 2
learning_steps_per_epoch = 2000
replay_memory_size = 1000

# NN learning settings
batch_size = 32

# Training regime
test_episodes_per_epoch = 100

# Other parameters
frame_repeat = 12
resolution = (128, 128)
episodes_to_watch = 10
doom_skill = 5

model_savefile = "./model.pth"


# Uses GPU if available
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")


def update_config(game):
    game.set_labels_buffer_enabled(True)
    game.set_automap_buffer_enabled(True)
    game.set_automap_mode(vzd.AutomapMode.OBJECTS)
    game.set_automap_rotate(False)
    game.set_automap_render_textures(False)

    game.set_depth_buffer_enabled(True)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_render_hud(False)
    game.set_doom_skill(doom_skill)


def get_game_variables_for_reward(game):
    kills = game.game_variables[0]
    damage_taken = game.game_variables[5]
    damage_count = game.game_variables[4]
    armor = game.game_variables[7]
    weapon_ammo = game.game_variables[12]
    return kills, damage_taken, damage_count, armor, weapon_ammo


def get_reward(game, prev_kills, prev_damage_taken, prev_damage_count, prev_armor, prev_weapon_ammo):
    reward = 0
    if game.game_variables[8]:
        reward -= 1
    else:
        reward += 0.1

    did_kill = (game.game_variables[0]-prev_kills) > 0
    was_damage_taken = (game.game_variables[5]-prev_damage_taken) > 0
    did_damage = (game.game_variables[4]-prev_damage_count) > 0
    did_get_armor = (game.game_variables[7]-prev_armor) > 0
    did_shot = (game.game_variables[12]-prev_weapon_ammo) < 0
    reward += did_kill * 1
    reward -= was_damage_taken * 0.15
    reward += did_damage * 0.8
    reward += did_get_armor * 0.2

    if not did_damage:
      reward -= did_shot*0.05
    
    return reward


class ActorCriticNet(nn.Module):
    def __init__(self, num_actions):
        super(ActorCriticNet, self).__init__()
        self.conv_screen = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),
        )
        self.conv_label = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),
        )
        self.conv_minimap = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),
        )
        self.conv_depth = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(2048*4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(23, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(128+256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.actor = nn.Linear(256, num_actions)
        self.critic = nn.Linear(256, 1)
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()

    def forward(self, screen, label, minimap, depth, numerical_data):
        x1 = self.conv_screen(screen)
        x2 = self.conv_label(label)
        x3 = self.conv_minimap(minimap)
        x4 = self.conv_depth(depth)
        x = self.fc1(self.flatten(torch.concat([x1, x2, x3, x4], dim=-1)))

        x2 = self.fc2(numerical_data)
        x = self.fc3(torch.concat([x, x2], dim=-1))
    
        return self.actor(x), self.critic(x)




class Agent:
    def __init__(self, actions, lr=learning_rate, gamma=discount_factor, load_model=False, 
                 epsilon=1.0, epsilon_decay=0.9996, epsilon_min=0.1, clip_epsilon=0.2, value_loss_coef=0.5, entropy_coef=0.01):
        self.actions = actions
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.lamda = 0.95
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.actor_net = ActorCriticNet(len(actions)).to(DEVICE)

        self.optimizer = optim.Adam(self.actor_net.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.5)

        if load_model:
            print("Loading model from: ", model_savefile)
            self.actor_net.load_state_dict(torch.load(model_savefile, map_location=torch.device('cpu')))

        self.memory = deque(maxlen=replay_memory_size)
        self.actor_net.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def preprocess(self, state):
        """Down samples image to resolution"""
        data = []

        if state is None:
            for _ in range(4):
                data.append(torch.zeros(1, *resolution))
            data.append(torch.zeros(23))
            return data
        
        if state.screen_buffer.shape != 3:
            data.append(torch.unsqueeze(torch.from_numpy(skimage.transform.resize(state.screen_buffer, resolution)), 0)/255.)
        else:
            data.append(torch.from_numpy(skimage.transform.resize(state.screen_buffer, resolution))/255.)

        if state.labels_buffer.shape != 3:
            data.append(torch.unsqueeze(torch.from_numpy(skimage.transform.resize(state.labels_buffer, resolution)), 0)/255.)
        else:
            data.append(torch.from_numpy(skimage.transform.resize(state.labels_buffer, resolution))/255.)

        if state.automap_buffer.shape != 3:
            data.append(torch.unsqueeze(torch.from_numpy(skimage.transform.resize(state.automap_buffer, resolution)), 0)/255.)
        else:
            data.append(torch.from_numpy(skimage.transform.resize(state.automap_buffer, resolution))/255.)

        if state.depth_buffer.shape != 3:
            data.append(torch.unsqueeze(torch.from_numpy(skimage.transform.resize(state.depth_buffer, resolution)), 0)/255.)
        else:
            data.append(torch.from_numpy(skimage.transform.resize(state.depth_buffer, resolution))/255.)
        
        numerical_data = np.array(np.concatenate([state.game_variables[6:8], state.game_variables[9:]])) / 250.
        numerical_data = torch.clip(torch.from_numpy(numerical_data), 0, 1)

        for i in range(4):
            data[i] = (data[i] - 0.5) * 2  
        
        numerical_data = torch.clip((numerical_data - 0.5) * 2, -1, 1)
        data.append(numerical_data)

        return data


    def choose_action(self, game_state):
        state = self.preprocess(game_state)
        state = [torch.unsqueeze(s, 0).float().to(DEVICE) for s in state]
        with torch.no_grad():
            self.actor_net.eval()
            logits, _ = self.actor_net(*state)

        # print(logits.max(), logits.min())
        probs = torch.log_softmax(logits[0], dim=-1)
        action = torch.multinomial(probs.exp(), 1).item()
        return self.actions[action], action
    

    def append_memory(self, state, action, reward, next_state, done):
        state = self.preprocess(state)
        next_state = self.preprocess(next_state)
        action_index = action if isinstance(action, int) else action.argmax()
        self.memory.append((state, action_index, reward, next_state, done))


    def agent_train(self, game_state, action, next_game_state, done):
        if len(self.memory) < batch_size:
            variables = get_game_variables_for_reward(game_state)
            reward = get_reward(next_game_state, *variables)
            action = torch.tensor(action)
            self.append_memory(game_state, action, reward, next_game_state, done)
            return

        self.actor_net.train()

        # batch = list(sorted(self.memory, key=lambda x: x[2]))[:batch_size]
        batch = random.sample(self.memory, batch_size)
        screen_buffers_t, label_buffers_t, automap_buffers_t, depth_buffers_t, numerical_t = [], [], [], [], []
        screen_buffers_t_1, label_buffers_t_1, automap_buffers_t_1, depth_buffers_t_1, numerical_t_1 = [], [], [], [], []
        actions, rewards, dones = [], [], []


        for state, action, reward, next_state, done in batch:
            screen_buffers_t.append(state[0])
            label_buffers_t.append(state[1])
            automap_buffers_t.append(state[2])
            depth_buffers_t.append(state[3])
            numerical_t.append(state[4])
            
            screen_buffers_t_1.append(next_state[0])
            label_buffers_t_1.append(next_state[1])
            automap_buffers_t_1.append(next_state[2])
            depth_buffers_t_1.append(next_state[3])
            numerical_t_1.append(next_state[4])

            actions.append(action)
            rewards.append(reward)
            dones.append(done)

        screen_buffers_t = torch.stack(screen_buffers_t).float().to(DEVICE)
        label_buffers_t = torch.stack(label_buffers_t).float().to(DEVICE)
        automap_buffers_t = torch.stack(automap_buffers_t).float().to(DEVICE)
        depth_buffers_t = torch.stack(depth_buffers_t).float().to(DEVICE)
        numerical_t = torch.stack(numerical_t).float().to(DEVICE)
        
        screen_buffers_t_1 = torch.stack(screen_buffers_t_1).float().to(DEVICE)
        label_buffers_t_1 = torch.stack(label_buffers_t_1).float().to(DEVICE)
        automap_buffers_t_1 = torch.stack(automap_buffers_t_1).float().to(DEVICE)
        depth_buffers_t_1 = torch.stack(depth_buffers_t_1).float().to(DEVICE)
        numerical_t_1 = torch.stack(numerical_t_1).float().to(DEVICE)

        actions = torch.tensor(actions).long().to(DEVICE)
        actions = torch.tensor([action.argmax() for action in actions]).long().to(DEVICE)

        rewards = torch.tensor(rewards).float().to(DEVICE)

        not_dones = (1 - torch.tensor(dones).float()).to(DEVICE)

        self.optimizer.zero_grad()

        logits, values = self.actor_net(screen_buffers_t, label_buffers_t, automap_buffers_t, depth_buffers_t, numerical_t)
        _, next_values = self.actor_net(screen_buffers_t_1, label_buffers_t_1, automap_buffers_t_1, depth_buffers_t_1, numerical_t_1)

        advantages = []
        advantage = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * not_dones[t] - values[t]
            advantage = delta + self.gamma * self.lamda * not_dones[t] * advantage
            advantages.insert(0, advantage)
        advantages = torch.stack(advantages).to(DEVICE)
        returns = advantages + values

        log_probs = torch.log_softmax(logits, dim=-1)

        log_probs_act_taken = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)


        with torch.no_grad():
            old_logits, _ = self.actor_net(screen_buffers_t, label_buffers_t, automap_buffers_t, depth_buffers_t, numerical_t)
            old_log_probs = torch.log_softmax(old_logits, dim=-1)
            old_log_probs_act_taken = old_log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        ratios = torch.exp(log_probs_act_taken - old_log_probs_act_taken)

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = torch.nn.functional.mse_loss(values, returns)

        entropy_bonus = -log_probs.mean()

        loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_bonus

        l2_lambda = 1e-3
        l2_reg = sum(param.pow(2.0).sum() for param in self.actor_net.parameters())
        loss += l2_lambda * l2_reg

        self.optimizer.step()
        self.scheduler.step(loss)  # Update the learning rate based on the loss

        loss.backward()

        for param in self.actor_net.parameters():
            torch.nn.utils.clip_grad_value_(param, clip_value=1.0)

        # for name, param in self.actor_net.named_parameters():
            # if param.grad is not None:
                # print(f"Layer: {name}, Gradient magnitude: {param.grad.data.norm(2).item()}")
                # print(f"Layer: {name}, Weight magnitude: {param.data.norm(2).item()}")



    def save_weights(self, model_savefile):
        print("Saving the network weights to:", model_savefile)
        torch.save(self.actor_net.state_dict(), model_savefile)

    def load_weights(self, model_savefile):
        self.actor_net.load_state_dict(torch.load(model_savefile, map_location=torch.device(DEVICE)))
