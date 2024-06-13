import datetime
import numpy as np
import os
import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.env_checker import check_env
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


def test_hardware(rob: "HardwareRobobo"):
    raise NotImplementedError("Hardwareis not implemented yet")


class CoppeliaSimEnv(gym.Env):
    """
    Custom Environment that follows gym interface
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, rob: "SimulationRobobo", seed=None, verbose=0):
        super(CoppeliaSimEnv, self).__init__()
        self.rob = rob
        self.seed = seed
        self.verbose = verbose
        # 4 actions: forward, backward, left, right
        self.action_space = spaces.Discrete(4)
        low = np.zeros(8)  # 8 sensors, all readings start at 0
        high = np.full(8, np.inf)  # 8 sensors, maximum reading is infinity
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float64) # add last 3 states
        # shoudl descrie robot and env. like whee ]l pos speed everythng also goal
        self.previous_position = None
        self.observation = None
        # motor speeds
        self.left_motor_speed = None
        self.right_motor_speed = None
        # reward stuff
        self.s_trans = None
        self.s_rot = None
        self.v_sens = None
        self.v_sens = None
        self.reward = None

    def calculate_reward(self):
        # use sped and dur 
        # Get the current speeds of the left and right motors
        wheel_position = self.rob.read_wheels()
        print(f"Wheel position: {wheel_position}")
        self.left_motor_speed = wheel_position.wheel_speed_l
        self.right_motor_speed = wheel_position.wheel_speed_r
        # print(f"Left motor speed: {self.left_motor_speed}")
        # print(f"Right motor speed: {self.right_motor_speed}")
        print(f"Left motor speed: {self.rob.read_wheels().wheel_speed_l}")
        print(f"Right motor speed: {self.rob.read_wheels().wheel_speed_r}")
        # Calculate the translational speed
        self.s_trans = self.left_motor_speed + self.right_motor_speed
        # Calculate the rotational speed and normalize it between 0 and 1
        self.s_rot = abs(self.left_motor_speed - self.right_motor_speed) / max(
            self.left_motor_speed, self.right_motor_speed, np.finfo(float).eps
        )
        # Get the values of all proximity sensors
        # proximity_sensors = self.rob.read_irs()
        proximity_sensors = self.observation
        # Find the value of the proximity sensor closest to an obstacle and
        # normalize it between 0 and 1
        self.v_sens = np.amin(proximity_sensors) / max(
            np.amax(proximity_sensors), np.finfo(float).eps
        )
        # Calculate the reward make it between -1 and 1
        reward = self.s_trans * (1 - self.s_rot) * (1 - self.v_sens)

        print(
            f"s_trans {self.s_trans}\ns_rot {self.s_rot}\nv_sens {self.v_sens}\nReward {reward}"# use addition instead
        )

        return reward

    def step(self, action):
        speed = 50
        duration = 200
        # Map integer action to Direction instance
        action = Direction(action)
        if action == Direction.FRONT:  # forward
            self.rob.move_blocking(speed, speed, duration)
        elif action == Direction.BACK:  # backward
            self.rob.move_blocking(-speed, -speed, duration)
        elif action == Direction.LEFT:  # left
            self.rob.move_blocking(-speed, speed, duration)
        elif action == Direction.RIGHT:  # right
            self.rob.move_blocking(speed, -speed, duration)
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
            f"Action: {action}\nObservation: {self.observation}\n\
                    Reward: {self.reward}"
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


class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and
    logs them to TensorBoard.
    """

    def __init__(self, model, params):
        super(HParamCallback, self).__init__()
        self.model = model
        self.params = params

    def _on_training_start(self) -> None:
        # set position random
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
        for sensor_name, index in IR_SENSOR_INDICES.items():
            self.tb_formatter.writer.add_scalar(
                f"observations/{sensor_name}", observation[index], self.num_timesteps
            )
        self.tb_formatter.writer.add_scalar("rewards/env", reward, self.num_timesteps)

        # Calculate the average episode length if there are completed episodes
        if self.model.ep_info_buffer:
            ep_len_mean = np.mean([info["l"] for info in self.model.ep_info_buffer])
            self.tb_formatter.writer.add_scalar(
                "rollout/ep_len_mean", ep_len_mean, self.num_timesteps
            )

        return True


def train_and_run_model(rob, verbose=0):
    model_name = f"DQN-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    models_dir = "/root/results/models/"
    tensorboard_dir = "/root/results/tensorboard4/"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    def make_env():
        env = CoppeliaSimEnv(rob=rob)
        check_env(env)
        return env

    vec_env = make_vec_env(
        make_env, n_envs=1, monitor_dir=os.path.join(models_dir, "monitor")
    )

    DQN_PARAMS = dict(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        train_freq=16,
        gradient_steps=-1,
        gamma=0.99,
        exploration_fraction=0.5,
        exploration_final_eps=0.07,
        target_update_interval=10,
        learning_starts=50,
        buffer_size=10000,
        batch_size=128,
        learning_rate=4e-3,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=tensorboard_dir,
        seed=2,
    )

    if verbose:
        print("Creating and training model")

    model = DQN(**DQN_PARAMS)

    model.learn(
        total_timesteps=1000, # moree
        tb_log_name=model_name,
        callback=HParamCallback(model=model, params=DQN_PARAMS),
    ) # run this muktuioke time thus is one episode
    model.save(os.path.join(models_dir, model_name))
    if verbose:
        print("Training complete")
        print(f"Model saved to {os.path.join(models_dir, model_name)}")

    obs = vec_env.reset()
    n_steps = 100
    print(f"Running the trained model for {n_steps} steps")
    for i in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        if verbose:
            print("----" * 20)
            print(f"Step {i}")
            print(f"Action: {action}")
            print(f"Observation: {obs}\nReward: {reward}\nDone: {done}\nInfo: {info}")
        if done:
            break

    if verbose:
        print("Stopping simulation")
    rob.stop_simulation()


def run_all_actions(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        train_and_run_model(rob, verbose=1)
    elif isinstance(rob, HardwareRobobo):
        test_hardware(rob)
