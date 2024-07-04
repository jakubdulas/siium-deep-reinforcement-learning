#!/usr/bin/env python3

import itertools as it
import os
import random
from collections import deque
from time import sleep, time
import matplotlib.pyplot as plt

import numpy as np
import skimage.transform
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import cv2

import vizdoom as vzd

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
resolution = (300, 300)
episodes_to_watch = 10
doom_skill = 5

model_savefile = "./actor-critic.pth"
save_model = False
load_model = True
skip_learning = True

# Configuration file path
# config_file_path = os.path.join(vzd.scenarios_path, "deathmatch.cfg")
# config_file_path = os.path.join(vzd.scenarios_path, "basic.cfg")
config_file_path =  "basic_config.cfg"

# Uses GPU if available
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")


def get_game_variables_for_reward(game):
    kills = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
    damage_taken = game.get_game_variable(vzd.GameVariable.DAMAGE_TAKEN)
    damage_count = game.get_game_variable(vzd.GameVariable.DAMAGECOUNT)
    armor = game.get_game_variable(vzd.GameVariable.ARMOR)
    return kills, damage_taken, damage_count, armor

def get_reward(game, prev_kills, prev_damage_taken, prev_damage_count, prev_armor):
    reward = 0
    if game.is_player_dead():
        reward -= 50
    else:
        reward -= 2

    did_kill = (game.get_game_variable(vzd.GameVariable.KILLCOUNT)-prev_kills) > 0
    was_damage_taken = (game.get_game_variable(vzd.GameVariable.DAMAGE_TAKEN)-prev_damage_taken) > 0
    did_damage = (game.get_game_variable(vzd.GameVariable.DAMAGECOUNT)-prev_damage_count) > 0
    did_get_armor = (game.get_game_variable(vzd.GameVariable.ARMOR)-prev_armor) > 0


    reward += did_kill * 10
    reward -= was_damage_taken * 0.5 
    reward += did_damage * 2
    reward += did_get_armor * 3  
    return reward


def display_buffers(state):
    depth = state.depth_buffer
    if depth is not None:
        cv2.imshow("ViZDoom Depth Buffer", depth)
        print("depth", depth.shape)

    labels = state.labels_buffer
    if labels is not None:
        cv2.imshow("ViZDoom Labels Buffer", labels)
        print("labels", labels.shape)

    automap = state.automap_buffer
    if automap is not None:
        cv2.imshow("ViZDoom Map Buffer", automap)
        print("automap", automap.shape)

    sleep_time = 0.028
    cv2.waitKey(int(sleep_time * 1000))


def create_simple_game():
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_labels_buffer_enabled(True)
    game.set_automap_buffer_enabled(True)
    game.set_automap_mode(vzd.AutomapMode.OBJECTS)
    game.set_automap_rotate(False)
    game.set_automap_render_textures(False)

    game.set_depth_buffer_enabled(True)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_render_hud(False)
    game.set_doom_skill(doom_skill)
    game.init()
    print("Doom initialized.")

    return game

def increase_doom_skill():
    global doom_skill
    doom_skill += 1
    game.set_doom_skill(doom_skill)


def test(game, agent):
    """Runs a test_episodes_per_epoch episodes and prints the result"""
    print("\nTesting...")
    test_scores = []
    for test_episode in trange(test_episodes_per_epoch, leave=False):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            best_action_index = agent.get_action(state)

            game.make_action(actions[best_action_index], frame_repeat)
        r = game.get_total_reward()
        test_scores.append(r)

    test_scores = np.array(test_scores)
    print(
        "Results: mean: {:.1f} +/- {:.1f},".format(
            test_scores.mean(), test_scores.std()
        ),
        "min: %.1f" % test_scores.min(),
        "max: %.1f" % test_scores.max(),
    )


def run(game, agent, actions, num_epochs, frame_repeat, steps_per_epoch=2000):
    """
    Run num epochs of training episodes.
    Skip frame_repeat number of frames after each action.
    """

    start_time = time()
    train_scores = []

    for epoch in range(num_epochs):
        game.new_episode()
        train_episode_scores = []
        global_step = 0
        print(f"\nEpoch #{epoch + 1}")
        current_episode_score = 0

        for _ in trange(steps_per_epoch, leave=False):
            state = game.get_state()
            action = agent.get_action(state)
            prev_game_variables = get_game_variables_for_reward(game)
            reward = game.make_action(actions[action], frame_repeat)
            done = game.is_episode_finished()
            reward = get_reward(game, *prev_game_variables)
            current_episode_score += reward

            next_state = game.get_state()
            agent.append_memory(state, action, reward, next_state, done)

            if global_step > batch_size:
                agent.train()

            if done:
                train_episode_scores.append(current_episode_score)
                current_episode_score = 0
                game.new_episode()

            global_step += 1

        train_episode_scores = np.array(train_episode_scores)

        print(
            "Results: mean: {:.1f} +/- {:.1f},".format(
                train_episode_scores.mean(), train_episode_scores.std()
            ),
            "min: %.1f," % train_episode_scores.min(),
            "max: %.1f," % train_episode_scores.max(),
        )

        test(game, agent)
        train_scores.append(train_episode_scores.mean())
        if save_model:
            print("Saving the network weights to:", model_savefile)
            torch.save(agent.actor_net.state_dict(), model_savefile)
        print("Total elapsed time: %.2f minutes" % ((time() - start_time) / 60.0))


        increase_doom_skill()

    game.close()
    return agent, game


class ActorCriticNet(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(ActorCriticNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=3, stride=1, padding=1, bias=False),
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
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1056, 256),
            nn.ReLU(),
        )
        self.actor = nn.Linear(256, num_actions)
        self.critic = nn.Linear(256, 1)

    def forward(self, visual_data, numerical_data):
        x1 = self.conv(visual_data)
        x1 = x1.view(x1.size(0), -1)
        x2 = self.fc1(numerical_data)
        x = self.fc2(torch.concat([x1, x2], dim=-1))
        return self.actor(x), self.critic(x)


class ActorCriticAgent:
    def __init__(self, input_shape, action_size, lr, gamma, load_model, epsilon=1.0, epsilon_decay=0.9996, epsilon_min=0.1):
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma

        self.actor_net = ActorCriticNet(input_shape, action_size).to(DEVICE)
        self.optimizer = optim.Adam(self.actor_net.parameters(), lr=lr)

        if load_model:
            print("Loading model from: ", model_savefile)
            self.actor_net.load_state_dict(torch.load(model_savefile, map_location=torch.device('cpu')))

        self.memory = deque(maxlen=replay_memory_size)

    def preprocess(self, state):
        """Down samples image to resolution"""
        if state is None:
            return np.zeros((2, *resolution)), np.zeros(5)

        visual_data = np.stack([
            state.depth_buffer,
            state.labels_buffer,
            # state.automap_buffer
        ], axis=-1)
        visual_data = skimage.transform.resize(visual_data, resolution)
        visual_data = visual_data.astype(np.float32)
        visual_data = np.transpose(visual_data, [2, 0, 1])
        print(np.max(visual_data))

        # img = np.expand_dims(img, axis=0)
        numerical_data = np.array(state.game_variables)/100
        numerical_data = np.clip(numerical_data, 0, 1)
    
        return visual_data, numerical_data

    def get_action(self, state):
        if np.random.uniform() < self.epsilon:
            return random.choice(range(self.action_size))
        else:
            state = self.preprocess(state)
            state = np.expand_dims(state[0], axis=0), np.expand_dims(state[1], axis=0)
            state = torch.from_numpy(state[0]).float().to(DEVICE), torch.from_numpy(state[1]).float().to(DEVICE)
            logits, _ = self.actor_net(*state)
            action = torch.argmax(logits).item()
            return action

    def append_memory(self, state, action, reward, next_state, done):
        state = self.preprocess(state)
        next_state = self.preprocess(next_state)
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        
        visual_states = np.array([sample[0][0] for sample in batch]).astype(float)
        numerical_states = np.array([sample[0][1] for sample in batch]).astype(float)
        
        actions = np.array([sample[1] for sample in batch]).astype(int)
        rewards = np.array([sample[2] for sample in batch]).astype(float)
        
        visual_next_states = np.array([sample[3][0] for sample in batch]).astype(float)
        numerical_next_states = np.array([sample[3][1] for sample in batch]).astype(float)
        
        dones = np.array([sample[4] for sample in batch]).astype(bool)
        not_dones = ~dones

        visual_states = torch.from_numpy(visual_states).float().to(DEVICE)
        numerical_states = torch.from_numpy(numerical_states).float().to(DEVICE)
        actions = torch.from_numpy(actions).long().to(DEVICE)
        rewards = torch.from_numpy(rewards).float().to(DEVICE)
        visual_next_states = torch.from_numpy(visual_next_states).float().to(DEVICE)
        numerical_next_states = torch.from_numpy(numerical_next_states).float().to(DEVICE)
        dones = torch.from_numpy(dones).float().to(DEVICE)

        self.optimizer.zero_grad()

        logits, values = self.actor_net(visual_states, numerical_states)
        _, next_values = self.actor_net(visual_next_states, numerical_next_states)

        values = values.squeeze()
        next_values = next_values.squeeze()

        returns = rewards + self.gamma * next_values.detach() * torch.from_numpy(not_dones).float().to(DEVICE)
        advantages = returns - values

        action_log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        action_log_probs = action_log_probs[range(batch_size), actions]

        actor_loss = -(advantages.detach() * action_log_probs).mean()
        critic_loss = advantages.pow(2).mean()

        loss = actor_loss + critic_loss
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min



if __name__ == "__main__":
    # Initialize game and actions
    game = create_simple_game()
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    # Initialize our agent with the set parameters
    agent = ActorCriticAgent(
        input_shape=(2, *resolution),
        action_size=len(actions),
        lr=learning_rate,
        gamma=discount_factor,
        load_model=load_model,
    )

    # Run the training for the set number of epochs
    if not skip_learning:
        agent, game = run(
            game,
            agent,
            actions,
            num_epochs=train_epochs,
            frame_repeat=frame_repeat,
            steps_per_epoch=learning_steps_per_epoch,
        )

        print("======================================")
        print("Training finished. It's time to watch!")

    # Reinitialize the game with window visible
    game.close()
    game.set_window_visible(True)
    game.set_labels_buffer_enabled(True)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    game.init()

    agent.epsilon = 0

    for _ in range(episodes_to_watch):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            best_action_index = agent.get_action(state)
            # display_buffers(state)
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)

    cv2.destroyAllWindows()
