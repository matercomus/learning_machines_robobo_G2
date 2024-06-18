import time
import csv
import pandas as pd
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from robobo_interface import (
    IRobobo,
    SimulationRobobo,
    HardwareRobobo,
)


def run_all_actions(rob: IRobobo):
    env = train_env(rob)
    env.rob.play_simulation()
    env.training_loop()
    # env.run_trained_model()
    env.rob.stop_simulation()

def reward_function_2(speed_L, speed_R, irs: list, sensor_max=200):
    s_trans = abs(speed_L) + abs(speed_R)
    if speed_L * speed_R < 0:
        s_rot = abs(abs(speed_L) - abs(speed_R)) / max(abs(speed_L), abs(speed_R))
    else:
        s_rot = 0

    if max(irs) >= sensor_max:
        v_sens = 1
    else:
        v_sens = (max(irs) - min(irs)) / (sensor_max - min(irs))

    return s_trans * (1 - s_rot) * (1 - v_sens)

def reward_function(speed_L, speed_R, irs: list, normalizer=0.01):
    speed_L = speed_L * normalizer
    speed_R = speed_R * normalizer
    closest = max(irs)

    # Turning penalty
    if  speed_L * speed_R < 0 and speed_L == speed_R:
        penalty_turning = (speed_L + speed_R) * 0.5
    else:
        penalty_turning = 0

    # Bump penalty
    if closest == np.inf:
        penalty_bump = 0
    else:
        if closest >= 200:
            penalty_bump = 20
        else:
            penalty_bump = closest * normalizer * 0.5

    return speed_L + speed_R - penalty_turning - penalty_bump

# Define the neural network model
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        self.initialize_weights()
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2048)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.3
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.batch_size = 20

    def save_model(self, file_name):
        torch.save(self.model.state_dict(), file_name)

    def load_model(self, file_name):
        self.model.load_state_dict(torch.load(file_name))
        self.model.eval()  # Set the model to evaluation mode

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return (random.uniform(-100, 100), random.uniform(-100, 100))
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            act_values = self.model(state)
        act_values = act_values.squeeze().cpu().numpy()
        act_values = np.clip(act_values, -100, 100)
        print(act_values)
        return act_values
    
    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state in minibatch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            target = self.model(state).detach().numpy()

            Q_future = self.model(next_state).detach().max().item()
            target[:] = reward + self.gamma * Q_future

            # Debugging: Print target values
            print(f"target: {target}")

            target_f = self.model(state)
            self.optimizer.zero_grad()
            loss = self.criterion(target_f, torch.FloatTensor(target))
            # Debugging: Print loss value
            print(f"loss: {loss.item()}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
    # def replay(self):
    #     minibatch = random.sample(self.memory, self.batch_size)
    #     for state, action, reward, next_state in minibatch:
    #         state = torch.FloatTensor(state)
    #         next_state = torch.FloatTensor(next_state)
    #         target = self.model(state).detach().numpy()

    #         Q_future = self.model(next_state).detach().max().item()
    #         target[:] = reward + self.gamma * Q_future
                
    #         target_f = self.model(state)
    #         self.optimizer.zero_grad()
    #         loss = self.criterion(target_f, torch.FloatTensor(target))
    #         loss.backward()
    #         self.optimizer.step()
            
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay


class train_env():
    def __init__(self, rob):
        self.rob = rob
        self.state_size = 8 # Number of IR sensors
        self.action_size = 2  # Speed of left and right wheels
        self.agent = DQNAgent(self.state_size, self.action_size)
        self.csv_file = '/root/results/data.csv'

    def normalize_irs(self, state, sensor_max=200):
        min_irs = min(state)
        max_irs = max(state)
        return [((irs - min_irs) / (max_irs - min_irs)) if max_irs < sensor_max else 1 for irs in state]

    def step(self, state, time=200):
        action = self.agent.act(self.normalize_irs(state))
        self.rob.move_blocking(action[0], action[1], time)
        next_state = self.rob.read_irs()
        reward = reward_function_2(action[0], action[1], next_state)
        return next_state, action, reward

    def training_loop(self):
        print('Training started')
        for epoch in range(10):
            state = self.rob.read_irs()
            for _ in range(40):
                next_state, action, reward = self.step(state)
                self.agent.remember(state, action, reward, next_state)
                state = next_state
                
                # store data
                with open(self.csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([epoch, state, action, reward, next_state])

            if len(self.agent.memory) >= self.agent.batch_size:
                self.agent.replay()
            print(f'end of {epoch + 1} epoch')
            self.agent.save_model('/root/results/dqn_model.pth')
        
        # Save the model after training
        # self.agent.save_model('/root/results/dqn_model.pth')

    def run_trained_model(self, max_steps=200):
        self.agent.load_model('/root/results/dqn_model.pth')  # Load the trained model

        state = self.rob.read_irs()
        total_reward = 0
        for step in range(max_steps):
            action = self.agent.act(self.normalize_irs(state))  # Use the trained model to decide actions
            self.rob.move_blocking(action[0], action[1], 200)
            next_state = self.rob.read_irs()
            reward = reward_function_2(action[0], action[1], next_state)
            total_reward += reward

            # Optionally, print or log the state, action, and reward
            print(f'Step: {step}, State: {state}, Action: {action}, Reward: {reward}')
            state = next_state
