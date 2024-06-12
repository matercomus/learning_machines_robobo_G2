import time
import csv
import datetime
import pandas as pd
import numpy as np
import os
from stable_baselines3 import PPO
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

from enum import Enum
from robobo_interface import (
    IRobobo,
    SimulationRobobo,
    HardwareRobobo,
)


class Direction(Enum):
    FRONT = "front"
    BACK = "back"
    LEFT = "left"
    RIGHT = "right"


# Mapping of directions to their corresponding IR sensor names
DIRECTION_IR_MAP = {
    Direction.FRONT: ["FrontL", "FrontR", "FrontC"],
    Direction.BACK: ["BackL", "BackR", "BackC"],
    Direction.LEFT: ["FrontLL", "FrontL"],
    Direction.RIGHT: ["FrontR", "FrontRR"],
}

# Mapping of IR sensor names to their indices
IR_SENSOR_INDICES = {
    "FrontL": 2,
    "FrontR": 3,
    "FrontC": 4,
    "BackL": 0,
    "BackR": 1,
    "BackC": 6,
    "FrontLL": 7,
    "FrontR": 3,
    "FrontRR": 5,
}


def write_data(data, time, irs, direction=None, event=None, run_id=0):
    """Write sensor data to the data dictionary."""
    data["time"].append(time_now(time))
    for sensor_name, sensor_index in IR_SENSOR_INDICES.items():
        data[sensor_name].append(irs[sensor_index])
    data["direction"].append(direction if direction is not None else "N/A")
    data["event"].append(event if event is not None else "N/A")
    data["run_id"].append(run_id)


def time_now(start_time):
    """Get the current time elapsed since start_time in milliseconds."""
    return round((time.time() - start_time) * 1000, 0)


def save_data_to_csv(data, mode, n_runs, timestamp, i):
    """Convert data to DataFrame and save to CSV."""
    df = pd.DataFrame(data)
    df.to_csv(
        f"/root/results/data/{mode}_{n_runs}_runs_{timestamp}.csv",
        mode="a",
        header=i == 0,
        index=False,
    )


def test_hardware(rob: "HardwareRobobo", mode="HW"):
    data = {
        "time": [],
        "FrontL": [],
        "FrontR": [],
        "FrontC": [],
        "BackL": [],
        "BackR": [],
        "BackC": [],
        "FrontLL": [],
        "FrontRR": [],
        "direction": [],
        "event": [],
        "run_id": [],
    }
    test(rob, mode=mode, ir_threshold=50, data=data, run_id=0)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_data_to_csv(data, mode, 1, timestamp, 0)


def test_simulation(rob: "SimulationRobobo", mode="SIM", n_runs=5):
    os.makedirs("/root/results/data", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for i in range(n_runs):
        # Reset data dictionary for each run
        data = {
            "time": [],
            "FrontL": [],
            "FrontR": [],
            "FrontC": [],
            "BackL": [],
            "BackR": [],
            "BackC": [],
            "FrontLL": [],
            "FrontRR": [],
            "direction": [],
            "event": [],
            "run_id": [],
        }

        rob.play_simulation()
        data = test(rob, run_id=i, data=data, mode=mode, ir_threshold=200)
        rob.stop_simulation()

        save_data_to_csv(data, mode, n_runs, timestamp, i)


def test(
    rob,
    run_id,
    mode,
    data,
    ir_threshold,
    speed: int = 50,
    move_duration: int = 200,
    turn_duration: int = 230,
):
    print("Running test in", mode, "mode")
    start_time = time.time()

    write_data(
        data=data,
        time=start_time,
        irs=rob.read_irs(),
        direction="forward",
        run_id=run_id,
    )

    while (
        rob.read_irs()[IR_SENSOR_INDICES["FrontL"]] < ir_threshold
        and rob.read_irs()[IR_SENSOR_INDICES["FrontL"]] != float("inf")
        and rob.read_irs()[IR_SENSOR_INDICES["FrontC"]] < ir_threshold
        and rob.read_irs()[IR_SENSOR_INDICES["FrontC"]] != float("inf")
        and rob.read_irs()[IR_SENSOR_INDICES["FrontR"]] < ir_threshold
        and rob.read_irs()[IR_SENSOR_INDICES["FrontR"]] != float("inf")
    ):
        write_data(
            data=data,
            time=start_time,
            irs=rob.read_irs(),
            direction="forward",
            run_id=run_id,
        )
        rob.move_blocking(speed, speed, move_duration)

    write_data(
        data=data,
        time=start_time,
        irs=rob.read_irs(),
        direction="forward",
        event="obstacle",
        run_id=run_id,
    )

    for _ in range(3):
        rob.move_blocking(-speed, -speed, move_duration)
        write_data(
            data=data,
            time=start_time,
            irs=rob.read_irs(),
            direction="backward",
            run_id=run_id,
        )

    for _ in range(5):
        rob.move_blocking(speed, -speed, turn_duration)
        write_data(
            data=data,
            time=start_time,
            irs=rob.read_irs(),
            direction="right",
            run_id=run_id,
        )

    for _ in range(10):
        rob.move_blocking(speed, speed, move_duration)
        write_data(
            data=data,
            time=start_time,
            irs=rob.read_irs(),
            direction="forward",
            run_id=run_id,
        )

    return data


def test_stable_baselines():
    model = PPO("MlpPolicy", "CartPole-v1", verbose=1).learn(1000)


def run_all_actions(rob: IRobobo):
    # if isinstance(rob, SimulationRobobo):
    #     # test_simulation(rob)
    #     test_stable_baselines()
    # elif isinstance(rob, HardwareRobobo):
    #     test_hardware(rob)
    env = train_env(rob)
    env.rob.play_simulation()
    # env.training_loop()
    env.run_trained_model()
    env.rob.stop_simulation()

def reward_function_2(speed_L, speed_R, irs: list, sensor_max=200):
    s_trans = speed_L + speed_R
    s_rot = abs(speed_L - speed_R) / max(speed_L, speed_R)

    if max(irs) >= sensor_max:
        v_sens = 1
    else:
        v_sens = (max(irs) - min(irs)) / (sensor_max - min(irs))

    return s_trans * (1 - s_rot) * (1 - v_sens)



def reward_function(speed_L, speed_R, irs: list, normalizer = 0.01):
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
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.batch_size = 32

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
        state = torch.FloatTensor(state)
        act_values = self.model(state)
        return act_values.detach().numpy()
        
    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state in minibatch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            target = self.model(state).detach().numpy()

            Q_future = self.model(next_state).detach().max().item()
            target[:] = reward + self.gamma * Q_future
                
            target_f = self.model(state)
            self.optimizer.zero_grad()
            loss = self.criterion(target_f, torch.FloatTensor(target))
            loss.backward()
            self.optimizer.step()
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class train_env():
    def __init__(self, rob):
        self.rob = rob
        self.state_size = 8 # Number of IR sensors
        self.action_size = 2  # Speed of left and right wheels
        self.agent = DQNAgent(self.state_size, self.action_size)
        self.csv_file = '/root/results/data.csv'

    def get_state(self):
        ir_readings = self.rob.read_irs()
        return ir_readings
    
    def step(self, state, time = 200):
        action = self.agent.act(state)
        self.rob.move_blocking(action[0], action[1], time)
        next_state = self.rob.read_irs()
        reward = reward_function_2(action[0], action[1], next_state)
        return next_state, action, reward

    def training_loop(self):
        for epoch in range(50):
            state = self.get_state()
            for _ in range(96):
                next_state, action, reward = self.step(state)
                self.agent.remember(state, action, reward, next_state)
                state = next_state
                
                # store data
                with open(self.csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([epoch, state, action, reward, next_state])

            if len(self.agent.memory) > self.agent.batch_size:
                self.agent.replay()
            print(f'end of {epoch} epoch')
        
        # Save the model after training
        self.agent.save_model('/root/results/dqn_model.pth')

    def run_trained_model(self, max_steps=200):
        self.agent.load_model('/root/results/dqn_model.pth')  # Load the trained model

        state = self.get_state()
        total_reward = 0
        for step in range(max_steps):
            action = self.agent.act(state)  # Use the trained model to decide actions
            self.rob.move_blocking(action[0], action[1], 200)
            next_state = self.get_state()
            reward = reward_function_2(action[0], action[1], next_state)
            total_reward += reward

            # Optionally, print or log the state, action, and reward
            print(f'Step: {step}, State: {state}, Action: {action}, Reward: {reward}')
            state = next_state
            
