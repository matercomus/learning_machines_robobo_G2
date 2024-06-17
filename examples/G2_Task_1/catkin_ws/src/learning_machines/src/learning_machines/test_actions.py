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

# idea: use camera separate it int lrf and calc pxels to detect objects or obstacles?


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
        # Multiply by 4 to include the current state and the last three states
        low = np.full(12 * 4, -np.inf)  # Allow negative values for all features
        high = np.full(12 * 4, np.inf)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float64)

        # Initialize the past observations to zeros
        self.past_observations = [np.zeros(12) for _ in range(3)]

        # Vars
        self.observation = None
        self.left_motor_speed = 0
        self.right_motor_speed = 0
        self.previous_position = None
        self.left_wheel_pos = 0
        self.previous_left_wheel_pos = 0
        self.right_wheel_pos = 0
        self.previous_right_wheel_pos = 0
        self.speed = 0
        self.duration = 0

        # Reward stuff
        self.s_trans = 0
        self.s_rot = 0
        self.v_sens = 0
        self.reward = 0
        self.action = None
        self.actions = []
        self.last_actions = []
        self.action_sequence_length = 5
        # Exploration reward stuff
        self.f_exp = 0
        self.grid_size = (1000, 1000)
        self.grid = np.zeros(self.grid_size)

    def calculate_speed(self, duration):
        # Get the current positions of the left and right wheels
        wheel_position = self.rob.read_wheels()
        current_left_wheel_pos = wheel_position.wheel_pos_l
        current_right_wheel_pos = wheel_position.wheel_pos_r

        # Calculate the speeds of the left and right wheels
        left_motor_speed = (
            current_left_wheel_pos - self.previous_left_wheel_pos
        ) / duration
        right_motor_speed = (
            current_right_wheel_pos - self.previous_right_wheel_pos
        ) / duration

        # Update the previous wheel positions
        self.previous_left_wheel_pos = current_left_wheel_pos
        self.previous_right_wheel_pos = current_right_wheel_pos

        # Calculate the translational speed
        speed = left_motor_speed + right_motor_speed

        return speed

    def calculate_reward(self, duration=200):
        # Get the current positions of the left and right wheels
        wheel_position = self.rob.read_wheels()
        current_left_wheel_pos = wheel_position.wheel_pos_l
        current_right_wheel_pos = wheel_position.wheel_pos_r

        # idea: reward check left ir vals and if going left penalize?

        # Calculate the speeds of the left and right wheels
        self.left_motor_speed = (
            current_left_wheel_pos - self.previous_left_wheel_pos
        ) / duration
        self.right_motor_speed = (
            current_right_wheel_pos - self.previous_right_wheel_pos
        ) / duration

        # Update the previous wheel positions
        self.previous_left_wheel_pos = current_left_wheel_pos
        self.previous_right_wheel_pos = current_right_wheel_pos

        # Calculate the translational speed
        self.s_trans = self.left_motor_speed + self.right_motor_speed

        # Calculate the rotational speed
        self.s_rot = abs(self.left_motor_speed - self.right_motor_speed)

        # Get the values of all proximity sensors
        proximity_sensors = np.array(self.rob.read_irs())

        # Set a threshold for the IR readings
        ir_threshold = 100

        # Penalize the reward if any IR reading is over the threshold
        self.v_sens = np.sum(proximity_sensors > ir_threshold)

        # Exploration factor
        self.f_exp = (
            np.sum(self.grid) / (self.grid_size[0] * self.grid_size[1])
        ) * 100000

        # Define weights for each factor
        w_trans = 0.2  # weight for translational speed
        w_rot = 0.1  # weight for rotational speed
        w_ir = 0.5  # weight for IR penalty
        w_exp = 0.2  # weight for exploration

        # Calculate the reward as a weighted sum of the factors
        reward = (
            w_trans * self.s_trans
            - w_rot * self.s_rot
            - w_ir * self.v_sens
            + w_exp * self.f_exp
        )

        print("----" * 10)
        print(
            f"Reward: {reward}\nTranslational speed: {self.s_trans}\nRotational\
            speed: {self.s_rot}\nIR penalty: {self.v_sens}\nExploration factor:\
            {self.f_exp}"
        )
        return reward

    def step(self, action):
        speed = 50
        duration = 300
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

        self.action = action
        self.actions.append(action)

        if isinstance(self.rob, SimulationRobobo) and not self.rob.is_running():
            raise RuntimeError("Simulation is not running")
        self.observation = np.array(self.rob.read_irs())
        self.observation = np.nan_to_num(self.observation, posinf=1e10)
        # Update the wheel positions and duration
        wheel_position = self.rob.read_wheels()
        self.left_wheel_pos = wheel_position.wheel_pos_l
        self.right_wheel_pos = wheel_position.wheel_pos_r
        self.duration = duration

        # Exploration reward stuff
        x, y = self.rob.get_position().x, self.rob.get_position().y
        cell_x, cell_y = int(x), int(y)
        self.grid[cell_x][cell_y] = 1

        # Update the observation with the new features
        self.observation = np.concatenate(
            [
                self.observation,
                np.array(
                    [
                        self.left_wheel_pos,
                        self.right_wheel_pos,
                        self.speed,
                        self.duration,
                    ]
                ),
            ]
        )

        self.reward = self.calculate_reward(duration=duration)
        # update last actions
        if len(self.actions) > self.action_sequence_length:
            self.last_actions = self.actions[-self.action_sequence_length :]

        terminated = False
        truncated = False
        info = {}

        # Update the past observations
        self.past_observations.pop(0)
        self.past_observations.append(self.observation)
        self.observation = np.concatenate(self.past_observations + [self.observation])

        if self.verbose:
            print("----" * 20)
            print("Step")
            print(
                f"Action: {action}\nObservation: {self.observation}\nReward: {self.reward}"
            )

        return (
            self.observation,
            self.reward,
            terminated,
            truncated,
            info,
        )

    def reset(self, seed=None, options=None):
        if isinstance(self.rob, SimulationRobobo):
            self.rob.stop_simulation()
            self.rob.play_simulation()
        if isinstance(self.rob, SimulationRobobo) and not self.rob.is_running():
            raise RuntimeError("Simulation is not running")
        observation = np.array(self.rob.read_irs())
        observation = np.nan_to_num(observation, posinf=1e10)

        # Update the wheel positions and duration
        wheel_position = self.rob.read_wheels()
        self.left_wheel_pos = wheel_position.wheel_pos_l
        self.right_wheel_pos = wheel_position.wheel_pos_r
        self.duration = 0  # Reset the duration to 0

        # Update the observation with the new features
        observation = np.concatenate(
            [
                observation,
                np.array(
                    [
                        self.left_wheel_pos,
                        self.right_wheel_pos,
                        self.speed,
                        self.duration,
                    ]
                ),
            ]
        )

        info = {}
        # Reset the past observations to zeros
        self.past_observations = [np.zeros(12) for _ in range(3)]
        # Reset exploration grid
        self.grid = np.zeros(self.grid_size)

        # Include the past observations in the returned observation
        return np.concatenate(self.past_observations + [observation]), info

    def close(self):
        if isinstance(self.rob, SimulationRobobo):
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
        self.actions = []

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

    def _log_observations(self):
        observation = np.array(self.training_env.envs[0].observation)
        left_motor_speed = np.array(self.training_env.envs[0].left_motor_speed)
        right_motor_speed = np.array(self.training_env.envs[0].right_motor_speed)

        for sensor_name, index in IR_SENSOR_INDICES.items():
            self.tb_formatter.writer.add_scalar(
                f"observations/{sensor_name}", observation[index], self.num_timesteps
            )
        self.tb_formatter.writer.add_scalar(
            "observations/left_motor_speed", left_motor_speed, self.num_timesteps
        )
        self.tb_formatter.writer.add_scalar(
            "observations/right_motor_speed", right_motor_speed, self.num_timesteps
        )

    def _log_rewards(self):
        s_trans = np.array(self.training_env.envs[0].s_trans)
        s_rot = np.array(self.training_env.envs[0].s_rot)
        v_sens = np.array(self.training_env.envs[0].v_sens)
        f_exp = np.array(self.training_env.envs[0].f_exp)
        reward = np.array(self.training_env.envs[0].reward)

        self.tb_formatter.writer.add_scalar(
            "rewards/s_trans", s_trans, self.num_timesteps
        )
        self.tb_formatter.writer.add_scalar("rewards/s_rot", s_rot, self.num_timesteps)
        self.tb_formatter.writer.add_scalar(
            "rewards/v_sens", v_sens, self.num_timesteps
        )
        self.tb_formatter.writer.add_scalar("rewards/f_exp", f_exp, self.num_timesteps)
        self.tb_formatter.writer.add_scalar("rewards/env", reward, self.num_timesteps)

    def _log_action(self):
        action = self.training_env.envs[0].action
        # Convert the Direction enum to its integer value
        action_value = action.value if isinstance(action, Direction) else action
        self.tb_formatter.writer.add_scalar(
            "actions/action", action_value, self.num_timesteps
        )
        # Add the action to the list of actions
        self.actions.append(action_value)
        # Log a histogram of actions
        self.tb_formatter.writer.add_histogram(
            "actions/histogram", np.array(self.actions), self.num_timesteps
        )

    def _log_episode_length(self):
        if self.model.ep_info_buffer:
            ep_len_mean = np.mean([info["l"] for info in self.model.ep_info_buffer])
            self.tb_formatter.writer.add_scalar(
                "rollout/ep_len_mean", ep_len_mean, self.num_timesteps
            )

    def _on_step(self) -> bool:
        self._log_observations()
        self._log_rewards()
        self._log_action()
        self._log_episode_length()

        return True


model_dir = "/root/results/models/"
tensorboard_dir = "/root/results/tensorboard5/"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(tensorboard_dir, exist_ok=True)


def train_model(
    rob,
    n_episodes=50,
    time_steps_per_episode=100,
    load_model=False,
    model_name=None,
    n_envs=1,
    verbose=0,
):
    def make_env():
        env = CoppeliaSimEnv(rob=rob)
        check_env(env)
        return env

    vec_env = make_vec_env(
        make_env, n_envs=n_envs, monitor_dir=os.path.join(model_dir, "monitor")
    )
    DQN_PARAMS = dict(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        train_freq=16,
        gradient_steps=-1,
        gamma=0.99,
        exploration_fraction=0.3,
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
    if load_model:
        if model_name is None:
            raise ValueError("model_name must be provided if load_model is True")
        model_path = os.path.join(model_dir, model_name)
        if verbose:
            print(f"Loading model from {model_path}")
        model = DQN.load(model_path, env=vec_env)
    else:
        model_name = f"DQN-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

        if verbose:
            print("Creating and training model")

        model = DQN(**DQN_PARAMS)

    print(f"Training model {model_name}")
    for episode in range(n_episodes):
        print(f"Episode {episode}")
        model.learn(
            total_timesteps=time_steps_per_episode,
            tb_log_name=model_name,
            callback=HParamCallback(model=model, params=DQN_PARAMS),
        )
        model.save(os.path.join(model_dir, model_name))
        if verbose:
            print(f"Episode {episode} training complete")
            print(f"Model saved to {os.path.join(model_dir, model_name)}")

    rob.stop_simulation()

    return model


def run_model(rob, model=None, model_name=None, vec_env=None, n_steps=100, verbose=1):
    if model is None:
        if model_name is None:
            raise ValueError("Either a model or a model_name must be provided")
        model_path = os.path.join(model_dir, model_name)
        if verbose:
            print(f"Loading model from {model_path}")
        model = DQN.load(model_path)

    if vec_env is None:
        vec_env = make_vec_env(
            lambda: CoppeliaSimEnv(rob=rob),
            n_envs=2,
            monitor_dir=os.path.join(model_dir, "monitor"),
        )

    obs = vec_env.reset()
    for step in range(n_steps):
        print(f"Step {step}/{n_steps}")
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        print(f"Action: {action}")

    if verbose:
        print("Stopping simulation")
    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()


def train_and_run_model(rob, verbose=0):
    model = train_model(rob, verbose=verbose)
    vec_env = make_vec_env(
        lambda: CoppeliaSimEnv(rob=rob),
        n_envs=1,
        monitor_dir=os.path.join(model_dir, "monitor"),
    )
    run_model(rob, model, vec_env, verbose=verbose)


def test_hardware(rob: "HardwareRobobo", model_name: str, n_steps: int = None):
    # Load the model
    model_path = os.path.join(model_dir, model_name)
    model = DQN.load(model_path)

    # Create the environment
    vec_env = make_vec_env(
        lambda: CoppeliaSimEnv(rob=rob),
        n_envs=1,
        monitor_dir=os.path.join(model_dir, "monitor"),
    )

    # Reset the environment
    obs = vec_env.reset()

    # If n_steps is not provided, run indefinitely
    if n_steps is None:
        step = 0
        while True:
            print(f"Step {step}")
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            step += 1
            print(f"Action: {action}")
    else:
        for step in range(n_steps):
            print(f"Step {step}/{n_steps}")
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            print(f"Action: {action}")

    print("Stopping simulation")


def run_all_actions(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        # train_and_run_model(rob, verbose=1)
        train_model(
            rob,
            n_episodes=25,
            time_steps_per_episode=200,
            verbose=1,
            load_model=False,
            n_envs=4,
            # model_name="DQN-20240614-020415_easy_50ep1kts",
        )
        # run_model(
        #     rob, model_name="DQN-20240614-020415_easy_50ep1kts", n_steps=200, verbose=1
        # )
    elif isinstance(rob, HardwareRobobo):
        test_hardware(rob, "DQN-20240614-130305", n_steps=200)
