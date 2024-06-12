import time
import datetime
from typing import Dict
import pandas as pd
import numpy as np
import os
import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam, TensorBoardOutputFormat
from gymnasium import spaces
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
        self.observation = None
        self.reward = None

    def calculate_reward(self):
        # Get the current speeds of the left and right motors
        wheel_position = self.rob.read_wheels()
        left_motor_speed = wheel_position.wheel_speed_l
        right_motor_speed = wheel_position.wheel_speed_r
        # Calculate the translational speed
        s_trans = left_motor_speed + right_motor_speed
        # Calculate the rotational speed and normalize it between 0 and 1
        s_rot = abs(left_motor_speed - right_motor_speed) / max(
            left_motor_speed, right_motor_speed, np.finfo(float).eps
        )
        # Get the values of all proximity sensors
        proximity_sensors = self.rob.read_irs()
        # Find the value of the proximity sensor closest to an obstacle and
        # normalize it between 0 and 1
        v_sens = np.amin(proximity_sensors) / max(
            np.amax(proximity_sensors), np.finfo(float).eps
        )
        # Calculate the reward
        reward = s_trans * (1 - s_rot) * (1 - v_sens)

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

        if not self.rob.is_running():
            raise RuntimeError("Simulation is not running")

        self.observation = np.array(self.rob.read_irs())
        self.observation = np.nan_to_num(self.observation, posinf=1e10)
        self.reward = self.calculate_reward()
        terminated = False
        truncated = False
        info = {}

        print("----" * 20)
        print("Step")
        print(
            f"Action: {action}\nObservation: {self.observation}\nReward: {self.reward}"
        )
        return (self.observation, self.reward, terminated, truncated, info)

    def reset(self, seed=None, options=None):
        self.rob.stop_simulation()
        self.rob.play_simulation()
        if not self.rob.is_running():
            raise RuntimeError("Simulation is not running")
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


class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and
    logs them to TensorBoard.
    """

    def __init__(self, model, params):
        super(HParamCallback, self).__init__()
        self.model = model
        self.params = params

        print("HParamCallback")

    def _on_training_start(self) -> None:
        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        self.tb_formatter = next(
            formatter
            for formatter in output_formats
            if isinstance(formatter, TensorBoardOutputFormat)
        )
        hparam_dict = {
            k: str(v)
            for k, v in self.params.items()
            if isinstance(v, (int, float, str))
        }
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0.0,
        }
        self.tb_formatter.writer.add_hparams(hparam_dict, metric_dict)

    def _on_step(self) -> bool:
        # Access the observation and reward from the environment
        observation = np.array(self.training_env.envs[0].observation)
        reward = np.array(self.training_env.envs[0].reward)

        # Log each element of the observation array as a separate scalar
        for sensor_name, index in IR_SENSOR_INDICES.items():
            self.tb_formatter.writer.add_scalar(
                f"observations/{sensor_name}", observation[index], self.num_timesteps
            )

        # Log the reward to TensorBoard
        self.tb_formatter.writer.add_scalar("rewards/env", reward, self.num_timesteps)

        return True


def train_and_run_model(rob):
    """This function trains an agent using the DQN algorithm and runs it in the
    simulation."""
    model_name = f"DQN-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    models_dir = "/root/results/models/"
    tensorboard_dir = "/root/results/tensorboard4/"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    def make_env():
        env = CoppeliaSimEnv(rob=rob)
        env = Monitor(env, filename=os.path.join(models_dir, "monitor", model_name))
        return env

    vec_env = make_vec_env(make_env, n_envs=1)

    DQN_PARAMS = dict(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        train_freq=16,
        gradient_steps=8,
        gamma=0.99,
        exploration_fraction=0.5,
        exploration_final_eps=0.07,
        target_update_interval=64,
        learning_starts=50,
        buffer_size=10000,
        batch_size=128,
        learning_rate=4e-3,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=tensorboard_dir,
        seed=2,
    )

    print("Creating and training model")
    model = DQN(**DQN_PARAMS)

    print("Training model")
    model.learn(
        total_timesteps=1000,
        tb_log_name=model_name,
        log_interval=10,
        callback=HParamCallback(model=model, params=DQN_PARAMS),
    )
    print("Training complete")
    model.save(os.path.join(models_dir, model_name))
    print(f"Model saved to {os.path.join(models_dir, model_name)}")

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

    print("Stopping simulation")
    rob.stop_simulation()


def run_all_actions(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        train_and_run_model(rob)
    elif isinstance(rob, HardwareRobobo):
        test_hardware(rob)
