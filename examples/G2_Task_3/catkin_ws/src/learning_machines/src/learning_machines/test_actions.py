import time
import os
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
    if isinstance(rob, SimulationRobobo):
        env.rob.play_simulation()
    # env.training_loop()
    env.run_trained_model()
    if isinstance(rob, SimulationRobobo):
        env.rob.stop_simulation()


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
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.batch_size = 128

    def save_model(self, file_name, epoch):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'epoch': epoch,
            'epsilon': self.epsilon
        }
        torch.save(checkpoint, file_name)

    def load_model(self, file_name):
        checkpoint = torch.load(file_name)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()  # Set the model to evaluation mode
        self.epsilon = checkpoint['epsilon']
        return checkpoint['epoch']

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
        for _ in range(64):
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
        self.state_size = 18  # Number of Feautures
        self.action_size = 3  # Three discrete actions: Forward, Left, Right
        self.agent = DQNAgent(self.state_size, self.action_size)
        self.csv_file = "/root/results/data.csv"
        self.run_name = f"test_run_{time.strftime('%Y%m%d-%H%M%S')}"
        self.IMG_SAVE_DIR = "/root/results/images/"
        self.img_id = 0

        # State values
        self.action = 0
        self.past_actions = []
        self.ir_readings = []
        self.green_percent_cells = np.zeros(9)
        self.last_green_percent_cells = np.zeros(9)
        self.red_percent_cells = np.zeros(9)
        self.last_red_percent_cells = np.zeros(9)
        self.reward = 0
        self.task_flag: bool = False

        # Color ranges
        self.lower_green = np.array([40, 40, 40])
        self.upper_green = np.array([80, 255, 255])
        self.lower_red8 = np.array([0, 120, 70])
        self.upper_red8 = np.array([10, 255, 255])
        self.lower_red9 = np.array([170, 120, 70])
        self.upper_red9 = np.array([180, 255, 255])

    def values_reset(self):
        self.action = 0
        self.past_actions = []
        self.ir_readings = []
        self.green_percent_cells = np.zeros(9)
        self.last_green_percent_cells = np.zeros(9)
        self.red_percent_cells = np.zeros(9)
        self.last_red_percent_cells = np.zeros(9)
        self.reward = 0
        self.task_flag: bool = False

    def color_change(self) -> bool:
        if self.task_flag:
            return self.green_percent_cells
        else:
            return self.red_percent_cells

    def reward_function(self):
        target_color = self.color_change()
        # Image vertical columns
        left = (target_color[0], target_color[3], target_color[6])
        middle = (target_color[1], target_color[4], target_color[7])
        right = (target_color[2], target_color[5], target_color[8])

        left_max = max(left)
        middle_max = max(middle)
        right_max = max(right)

        reward = 0

        # Penalty for bump
        if max(target_color) == 0 and (
            self.ir_readings[4] == 1
            or self.ir_readings[2] == 1
            or self.ir_readings[3] == 1
        ):
            return -10
        # Penalty for turning left and right
        l_r = [1,2,1,2]
        r_l = [2,1,2,1]
        l_r_2 = [2,2,1,1]
        r_l_2 = [1,1,2,2]
        h = self.past_actions[-4:]
        if len(self.past_actions) >= 4 and (
            h == l_r or h == r_l or h == l_r_2 or h == r_l_2
            ):
            return -3
        # If all states are 0 keep turning
        if max(self.ir_readings) == 0 and max(target_color) == 0 and self.action == 2:
            return 1
        # Object in the middle
        if middle_max == 1 and left_max == 0 and right_max == 0:
            if self.action == 0:  # go forward
                if middle.index(middle_max) == 0:
                    reward = 1
                if middle.index(middle_max) == 1:
                    reward = 3
                if middle.index(middle_max) == 2:
                    reward = 5
        # Object on the left
        elif left_max > right_max:
            if self.action == 1:  # go left
                if left.index(left_max) == 0:
                    reward = 5
                if left.index(left_max) == 1:
                    reward = 3
                if left.index(left_max) == 2:
                    reward = 1
        # Object on the right
        elif left_max < right_max:
            if self.action == 2:  # go right
                if right.index(right_max) == 0:
                    reward = 5
                if right.index(right_max) == 1:
                    reward = 3
                if right.index(right_max) == 2:
                    reward = 1
        # Swich to objective 2 if object was close in the middle and action was forward
        if not self.task_flag and self.last_red_percent_cells[7] == 1 and max(target_color) == 0 and self.action == 0:
            self.task_flag = True
            return 10
        # Task finished!!!
        if isinstance(self.rob, SimulationRobobo):
            if self.rob.base_detects_food():
                return 10
        return reward

    def step(self, state, time=200):
        self.action = self.agent.act(state)
        if self.action == 0:  # Forward
            self.rob.move_blocking(50, 50, 300)
        elif self.action == 1:  # Left
            self.rob.move_blocking(-15, 15, time)
        elif self.action == 2:  # Right
            self.rob.move_blocking(15, -15, time)

        self.past_actions.append(self.action)
        self.ir_readings = self.read_discrete_irs()

        self.last_green_percent_cells = self.green_percent_cells
        self.last_red_percent_cells = self.red_percent_cells
        self.green_percent_cells, self.red_percent_cells = (
            self.get_image_green_red_percent_cells()
        )
        self.reward = self.reward_function()

        # action_array = np.array([self.action])
        flag_array = np.array([self.task_flag])
        if not self.task_flag:
            target_color = self.red_percent_cells
        else:
            target_color = self.green_percent_cells
        next_state = np.concatenate(
            [
                flag_array,
                self.ir_readings,
                target_color,
            ]
        )
        self.img_id += 1

        return next_state

    def read_discrete_irs(self):
        ir_readings = self.rob.read_irs()
        discrete_ir_readings = []
        top_ir_threshold = 200
        bottom_ir_threshold = 50
        for ir in ir_readings:
            # Normalize the IR readings to range (0, 1)
            normalized_ir = max(
                0,
                min(
                    1,
                    (ir - bottom_ir_threshold)
                    / (top_ir_threshold - bottom_ir_threshold),
                ),
            )
            discrete_ir = np.digitize(normalized_ir, np.linspace(0, 1, 11)) - 1
            # Convert digitized IR reading back to float in range 0.0 to 1.0
            float_ir = discrete_ir / 10.0
            discrete_ir_readings.append(float_ir)

        return np.array(discrete_ir_readings)

    def process_image(
        self,
        image,
        color,
        color_lower1,
        color_upper1,
        color_lower2=None,
        color_upper2=None,
        save_imgs=False,
    ):
        image = cv2.resize(image, (64, 64))
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv_image, color_lower1, color_upper1)
        if color_lower2 is not None and color_upper2 is not None:
            mask2 = cv2.inRange(hsv_image, color_lower2, color_upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = mask1
        processed_image = cv2.bitwise_and(image, image, mask=mask)

        if save_imgs and id is not None:
            cv2.imwrite(
                os.path.join(
                    self.IMG_SAVE_DIR,
                    f"{self.run_name}_{color}_processed_image_{self.img_id}.png",
                ),
                processed_image,
            )

        return processed_image

    @staticmethod
    def get_color_percent_per_cell(image):
        grid_size = 3
        cell_size = image.shape[0] // grid_size
        color_percent = []
        threshold = 0.00001
        for i in range(grid_size):
            for j in range(grid_size):
                cell = image[
                    i * cell_size : (i + 1) * cell_size,
                    j * cell_size : (j + 1) * cell_size,
                ]
                non_black_pixels = np.sum(cell != 0)
                total_pixels = cell_size * cell_size
                percent = round(non_black_pixels / total_pixels, 3)
                color_percent.append(1 if percent > threshold else 0)
        return color_percent

    def get_image_green_red_percent_cells(self, save_img=True):
        save_dir = os.path.join(
            self.IMG_SAVE_DIR,
            self.run_name,
            f"{self.run_name}_stiched_image_{self.img_id}.png",
        )
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        image = self.rob.get_image_front()
        green_image = self.process_image(
            image,
            "green",
            self.lower_green,
            self.upper_green,
        )
        red_image = self.process_image(
            image,
            "red",
            self.lower_red8,
            self.upper_red8,
            self.lower_red9,
            self.upper_red9,
        )
        if save_img:
            print(f"Saving images in {save_dir}")
            stitched_image = np.hstack(
                (
                    cv2.resize(image, (64, 64)),
                    green_image,
                    red_image,
                )
            )
            cv2.imwrite(
                save_dir,
                stitched_image,
            )

        green_percent_cells = np.array(self.get_color_percent_per_cell(green_image))
        red_percent_cells = np.array(self.get_color_percent_per_cell(red_image))
        return green_percent_cells, red_percent_cells

    def early_termination(self):
        if self.reward == -10:
            print("Early termination due to IR reading")
            return 1
        elif self.rob.base_detects_food():
            print("GOAL REACHED!!! No need for more training.")
            return 2
        return 0

    def training_loop(self, load_previous_model=True):
        start_epoch = 0
        if load_previous_model:
            start_epoch = self.agent.load_model("/root/results/dqn_model.pth")
        print("Training started")
        for epoch in range(start_epoch, start_epoch + 50):
            self.rob.stop_simulation()
            self.rob.play_simulation()
            self.rob.set_phone_tilt(109, 100)
            if epoch > 0:
                self.values_reset()
            self.ir_readings = self.read_discrete_irs()
            # action_array = np.array([self.action])
            flag_array = np.array([self.task_flag])
            if not self.task_flag:
                target_color = self.red_percent_cells
            else:
                target_color = self.green_percent_cells
            state = np.concatenate(
                [
                    flag_array,
                    self.ir_readings,
                    target_color,
                ]
            )
            for _ in range(128):
                next_state = self.step(state)
                self.agent.remember(state, self.action, self.reward, next_state)
                # store data
                with open(self.csv_file, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([epoch, self.task_flag, self.reward, self.action, self.ir_readings, self.red_percent_cells, self.green_percent_cells, self.img_id])

                state = next_state
                print("-" * 30)
                print("Epoch: ", epoch + 1)
                print("Task:", self.task_flag)
                print("IR readings: ", self.ir_readings)
                print("Action: ", self.action)
                print("Green percent cells: ", self.green_percent_cells)
                print("Red percent cells: ", self.red_percent_cells)
                print("Reward: ", self.reward)
                if self.early_termination() == 1:
                    break
                # Train the model last time and stop the training.
                elif self.early_termination() == 2:
                    if len(self.agent.memory) >= self.agent.batch_size:
                        self.agent.replay()
                    self.agent.save_model("/root/results/dqn_model_best.pth", epoch + 1)
                    if not load_previous_model:
                        exit(1)

            if len(self.agent.memory) >= self.agent.batch_size:
                self.agent.replay()
            print(f"end of {epoch + 1} epoch")
            self.agent.save_model("/root/results/dqn_model.pth", epoch + 1)

    def run_trained_model(self, max_steps=400):  # TODO update
        # Load the trained model
        if isinstance(self.rob, SimulationRobobo):
            self.rob.stop_simulation()
            self.rob.play_simulation()
        self.rob.set_phone_tilt(109, 100)
        self.agent.load_model("/root/results/dqn_model_best.pth")
        self.ir_readings = self.read_discrete_irs()
        flag_array = np.array([self.task_flag])
        if not self.task_flag:
            target_color = self.red_percent_cells
        else:
            target_color = self.green_percent_cells
        state = np.concatenate(
            [
                flag_array,
                self.ir_readings,
                target_color,
            ]
        )
        total_reward = 0
        for step in range(max_steps):
            # Use the trained model to decide actions
            next_state = self.step(state)
            reward = self.reward_function()
            total_reward += reward

            # Optionally, print or log the state, action, and reward
            print("-" * 30)
            print(f"Step: {step}")
            print("Task:", self.task_flag)
            print("IR readings: ", self.ir_readings)
            print("Action: ", self.action)
            print("Green percent cells: ", self.green_percent_cells)
            print("Red percent cells: ", self.red_percent_cells)
            print("Reward: ", self.reward)
            print(f"\nTotal Reward: {total_reward}\n")
            state = next_state
