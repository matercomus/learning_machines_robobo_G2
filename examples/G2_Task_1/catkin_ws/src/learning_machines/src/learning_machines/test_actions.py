import time
import datetime
import pandas as pd
import numpy as np
import os
import gymnasium as gym

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from enum import Enum
from robobo_interface import (
    IRobobo,
    SimulationRobobo,
    HardwareRobobo,
)


class Direction(Enum):
    FRONT = 0
    BACK = 1
    LEFT = 2
    RIGHT = 3


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


class CoppeliaSimEnv(gym.Env):
    """
    Custom Environment that follows gym interface
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, rob: "SimulationRobobo"):
        super(CoppeliaSimEnv, self).__init__()
        self.rob = rob
        # 4 actions: forward, backward, left, right
        self.action_space = spaces.Discrete(4)
        low = np.zeros(8)  # 8 sensors, all readings start at 0
        high = np.full(8, np.inf)  # 8 sensors, maximum reading is infinity
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float64)
        self.previous_position = None

    def calculate_reward(self):
        distance_threshold = 200
        speed_reward = 0.1  # Reward for moving fast
        obstacle_penalty = -1000  # Penalty for hitting an obstacle
        idle_penalty = -1  # Penalty for staying in one spot

        # Get current position
        current_position = self.rob.get_position()

        # Check if the agent hit an obstacle
        if any(self.rob.read_irs()) > distance_threshold:
            return obstacle_penalty

        # Check if the agent is moving
        if (
            self.previous_position is not None
            and current_position == self.previous_position
        ):
            return idle_penalty

        # Reward the agent for moving fast
        if self.previous_position is not None:
            speed = (
                (current_position.x - self.previous_position.x) ** 2
                + (current_position.y - self.previous_position.y) ** 2
            ) ** 0.5
            reward = speed * speed_reward
        else:
            reward = 0

        # Update previous position
        self.previous_position = current_position

        return reward

    def step(self, action):
        speed = 50
        # Map integer action to Direction instance
        action = Direction(action)
        # Execute one time step within the environment
        if action == Direction.FRONT:  # forward
            self.rob.move_blocking(speed, speed, 200)
        elif action == Direction.BACK:  # backward
            self.rob.move_blocking(-speed, -speed, 200)
        elif action == Direction.LEFT:  # left
            self.rob.move_blocking(-speed, speed, 200)
        elif action == Direction.RIGHT:  # right
            self.rob.move_blocking(speed, -speed, 200)
        else:
            raise ValueError(f"Invalid action {action}")

        observation = np.array(self.rob.read_irs())
        observation = np.nan_to_num(observation, posinf=1e10)
        reward = self.calculate_reward()
        terminated = False
        truncated = False
        info = {}

        return (observation, reward, terminated, truncated, info)

    def reset(self, seed=None, options=None):
        self.rob.stop_simulation()
        self.rob.play_simulation()
        observation = np.array(self.rob.read_irs())
        observation = np.nan_to_num(observation, posinf=1e10)
        info = {}
        return (observation, info)

    def close(self):
        self.rob.stop_simulation()


def test_stable_baselines(rob):
    env = CoppeliaSimEnv(rob=rob)
    check_env(env, warn=True)

    obs, _ = env.reset()
    print("Initial observation:", obs)
    print("Action space:", env.action_space)
    print("Action space sample:", env.action_space.sample())

    # Hardcoded go left
    action = Direction.LEFT
    n_steps = 100
    for i in range(n_steps):
        print("Step ", i)
        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        print(f"Observation: {observation}, Reward: {reward}, Done: {done}")
        if done:
            break


def train_and_run_model(rob):
    """This function trains an agent using the A2C algorithm and runs it in the
    simulation."""

    def make_env():
        return CoppeliaSimEnv(rob=rob)

    vec_env = make_vec_env(make_env, n_envs=1)
    print("Creating and training model")
    model = A2C("MlpPolicy", vec_env, verbose=1).learn(1000)
    print("Training complete")
    obs = vec_env.reset()
    n_steps = 100
    print(f"Running model for {n_steps} steps")
    for i in range(n_steps):
        print("----" * 20)
        print(f"Step {i}")
        action, _ = model.predict(obs, deterministic=True)
        print(f"Action: {action}")
        obs, reward, done, info = vec_env.step(action)
        print(f"Observation: {obs}\nReward: {reward}\nDone: {done}\nInfo: {info}")
        if done:
            break

    rob.stop_simulation()


def run_all_actions(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        train_and_run_model(rob)
    elif isinstance(rob, HardwareRobobo):
        test_hardware(rob)
