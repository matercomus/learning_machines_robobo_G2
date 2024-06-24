import cv2
import csv
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


# Define the neural network model
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_size)
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
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1  # exploration rate
        self.epsilon_min = 0.2
        self.epsilon_decay = 0.9
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.batch_size = 64

    def save_model(self, file_name):
        torch.save(self.model.state_dict(), file_name)

    def load_model(self, file_name):
        self.model.load_state_dict(torch.load(file_name))
        self.model.eval()  # Set the model to evaluation mode

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice([0, 1, 2])  # Forward, Left, Right
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            act_values = self.model(state)
        return np.argmax(act_values.squeeze().cpu().numpy())

    def replay(self):
        for _ in range(100):
            minibatch = random.sample(self.memory, self.batch_size)
            for state, action, reward, next_state in minibatch:
                state = torch.FloatTensor(state)
                next_state = torch.FloatTensor(next_state)
                target = self.model(state).detach().clone()

                Q_future = self.model(next_state).detach().max().item()
                target[action] = reward + self.gamma * Q_future

                target_f = self.model(state)
                self.optimizer.zero_grad()
                loss = self.criterion(target_f, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class train_env:
    def __init__(self, rob):
        self.rob = rob
        self.state_size = 13  # Number of Feautures
        self.action_size = 3  # Three discrete actions: Forward, Left, Right
        self.agent = DQNAgent(self.state_size, self.action_size)
        self.csv_file = "/root/results/data.csv"

        # State values
        self.action = 0
        self.ir_readings = []
        self.position_history = []
        self.green_percent = 0
        self.reward = 0
        self.last_reward = 0
        self.last_green_percent = 0
        self.collected_food = 0
        self.past_rewards = []

    def values_reset(self):
        self.action = 0
        self.ir_readings = []
        self.position_history = []
        self.green_percent = 0
        self.reward = 0
        self.last_reward = 0
        self.last_green_percent = 0
        self.collected_food = 0
        self.past_rewards = []

    def reward_function(self):
        # Big reward for collecting food
        if self.rob.nr_food_collected() > self.collected_food:
            print("\n FOOD COLLECTED \n")
            self.collected_food = self.rob.nr_food_collected()
            reward = 40
        # Same position penalty
        x, y = self.rob.get_position().x, self.rob.get_position().y
        x, y = round(x, 1), round(y, 1)
        print("\n position: ", x, y)
        if (x, y) in self.position_history:
            pos_p = 0.5
        else:
            pos_p = 0
        self.position_history.append((x, y))
        # IR readings penalty
        ir_p = 0
        highest_ir = max(self.ir_readings)
        if self.green_percent > 0:
            ir_p = 0
        elif highest_ir >= 1.0:
            ir_p = 40
        else:
            ir_p = highest_ir * 10
        # Reward
        reward = (self.green_percent * 10 - ir_p) * pos_p

        # Define the bin edges for digitization
        bins = np.linspace(-40, 40, 21)  # 21 edges for 20 bins
        # Digitize the reward
        digitized_reward = (
            np.digitize(reward, bins) - 1
        )  # Subtract 1 to make bins start from 0

        # Convert digitized reward back to float in range -1.0 to 1.0
        float_reward = digitized_reward / 10.0 - 1.0

        return float_reward

    def step(self, state, time=200):
        self.action = self.agent.act(state)
        if self.action == 0:  # Forward
            self.rob.move_blocking(50, 50, time)
        elif self.action == 1:  # Left
            self.rob.move_blocking(-15, 35, time)
        elif self.action == 2:  # Right
            self.rob.move_blocking(35, -15, time)

        self.ir_readings = self.read_discrete_irs()
        image = self.process_image(self.rob.get_image_front())
        self.last_green_percent = self.green_percent
        self.green_percent = self.get_green_percent(image[:, :, 1])
        self.last_reward = self.reward
        self.reward = self.reward_function()
        self.past_rewards.append(self.reward)

        next_state = np.concatenate(
            [
                self.ir_readings,
                np.array(
                    [
                        self.action,
                        self.green_percent,
                        self.last_green_percent,
                        self.reward,
                        self.last_reward,
                    ]
                ),
            ]
        )

        return next_state

    def read_discrete_irs(self):
        ir_readings = self.rob.read_irs()
        discrete_ir_readings = []
        top_ir_threshold = 200
        bottom_ir_threshold = 50
        for ir in ir_readings:
            # Normalize the IR readings to range (0, 1)
            normalized_ir = (ir - bottom_ir_threshold) / (
                top_ir_threshold - bottom_ir_threshold
            )
            discrete_ir = np.digitize(normalized_ir, np.linspace(0, 1, 11)) - 1
            # Convert digitized IR reading back to float in range 0.0 to 1.0
            float_ir = discrete_ir / 10.0
            discrete_ir_readings.append(float_ir)

        return discrete_ir_readings

    def process_image(self, image):
        # Resize the image to 64x64 pixels
        image = cv2.resize(image, (64, 64))
        # Flip the image back
        image = cv2.flip(image, 0)
        # Isolate green channel
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])
        # Mask the image
        mask = cv2.inRange(hsv_image, lower_green, upper_green)
        return cv2.bitwise_and(image, image, mask=mask)

    def get_green_percent(self, image):
        # Check if there are any 1s in the array
        if not np.any(image):
            return 0
        # Define the center of the arra
        center = np.array([32, 32])
        # Get the coordinates of all 1s in the array
        ones_positions = np.argwhere(image != 0)
        # Calculate the distance of each 1 from the center
        distances = np.linalg.norm(ones_positions - center, axis=1)
        # Calculate the score based on the distances
        # Closer to center gives higher score, farther gives lower score
        min_distance = np.min(distances)
        if min_distance == 0:
            return 1
        else:
            # Normalize distance to range (0, 1) and invert it
            normalized_distance = min_distance / np.linalg.norm(center)
            score = 1 - normalized_distance

        # Define the bin edges for digitization
        bins = np.linspace(0, 1, 11)  # 11 edges for 10 bins
        # Digitize the score
        digitized_score = (
            np.digitize(score, bins) - 1
        )  # Subtract 1 to make bins start from 0

        # Convert digitized score back to float in range 0.0 to 1.0
        float_score = digitized_score / 10.0

        return float_score

    def early_termination(self):
        if all(self.reward <= -5 for self.reward in self.past_rewards[-10:]):
            print("Early termination due to low rewards")
            return True
        return False

    def training_loop(self):
        print("Training started")
        for epoch in range(100):
            self.rob.stop_simulation()
            self.rob.play_simulation()
            self.rob.set_phone_tilt(109, 50)
            if epoch > 0:
                self.values_reset()
            self.ir_readings = self.read_discrete_irs()
            state = np.concatenate(
                [
                    self.ir_readings,
                    np.array(
                        [
                            self.action,
                            self.green_percent,
                            self.last_green_percent,
                            self.reward,
                            self.last_reward,
                        ]
                    ),
                ]
            )
            for _ in range(128):
                next_state = self.step(state)
                self.agent.remember(state, self.action, self.reward, next_state)
                # store data
                with open(self.csv_file, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([epoch, state, next_state])

                state = next_state
                print("Reward: ", self.reward)
                if self.early_termination():
                    break

            if len(self.agent.memory) >= self.agent.batch_size:
                self.agent.replay()
            print(f"end of {epoch + 1} epoch")
            self.agent.save_model("/root/results/dqn_model.pth")

    def run_trained_model(self, max_steps=200):
        # Load the trained model
        self.rob.stop_simulation()
        self.rob.play_simulation()
        self.rob.set_phone_tilt(109, 50)
        self.agent.load_model("/root/results/dqn_model.pth")
        self.ir_readings = self.read_discrete_irs()
        state = np.concatenate(
            [
                self.ir_readings,
                np.array(
                    [
                        self.action,
                        self.green_percent,
                        self.last_green_percent,
                        self.reward,
                        self.last_reward,
                    ]
                ),
            ]
        )
        state = self.read_discrete_irs()
        total_reward = 0
        for step in range(max_steps):
            # Use the trained model to decide actions
            next_state = self.step(state)
            reward = self.reward_function()
            total_reward += reward

            # Optionally, print or log the state, action, and reward
            print(
                f"Step: {step}, State: {state}, Action: {self.action}, Reward: {reward}\n"
            )
            print(f"Total Reward: {total_reward}")
            state = next_state
