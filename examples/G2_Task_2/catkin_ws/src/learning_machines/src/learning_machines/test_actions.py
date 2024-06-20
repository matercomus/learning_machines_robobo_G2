import datetime
import numpy as np
import os
import gymnasium as gym
import cv2
import time

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
model_dir = "/root/results/models/"
tensorboard_dir = "/root/results/tensorboard6/"
image_dir = "/root/results/images"
image_run_dir = os.path.join(image_dir, time.strftime("%Y%m%d-%H%M"))
os.makedirs(model_dir, exist_ok=True)
os.makedirs(tensorboard_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)
os.makedirs(image_run_dir, exist_ok=True)


class Direction(Enum):
    FRONT = 0
    # BACK = 1
    LEFT = 1
    RIGHT = 2


# Mapping of directions to their corresponding IR sensor names
DIRECTION_IR_MAP = {
    Direction.FRONT: ["FrontL", "FrontR", "FrontC"],
    # Direction.BACK: ["BackL", "BackR", "BackC"],
    Direction.LEFT: ["FrontLL", "FrontL"],
    Direction.RIGHT: ["FrontR", "FrontRR"],
}

# Mapping of IR sensor names to their indices
IR_SENSOR_INDICES = {
    "FrontL": 2,
    "FrontR": 3,
    "FrontC": 4,
    # "BackL": 0,
    # "BackR": 1,
    # "BackC": 6,
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
        self.action_space = spaces.Discrete(3)
        self.image_shape = (64, 64)
        # Define the observation space for the image
        self.image_space = spaces.Box(
            low=0, high=255, shape=self.image_shape, dtype=np.uint8
        )
        low = np.full(9, -np.inf)  # Allow negative values for all features
        high = np.full(9, np.inf)
        self.sensor_space = spaces.Box(low=low, high=high, dtype=np.float64)

        # Combine the sensor and image spaces into the observation space
        self.observation_space = spaces.Dict(
            {"vector": self.sensor_space, "image": self.image_space}
        )

        # Vars
        self.observation = None
        self.ir_readings = None
        self.left_motor_speed = 0
        self.right_motor_speed = 0
        self.previous_position = None
        self.left_wheel_pos = 0
        self.previous_left_wheel_pos = 0
        self.right_wheel_pos = 0
        self.previous_right_wheel_pos = 0
        self.speed = 0
        self.duration = 0
        self.image_counter = 0
        # Mask
        self.mask = np.ones(8, dtype=bool)
        exclude_indices = np.array([0, 1, 6])
        self.mask[exclude_indices] = False

        # Reward stuff
        self.s_trans = 0
        self.s_rot = 0
        self.v_sens = 0
        self.reward = 0
        self.past_rewards = []
        self.action = None
        self.actions = []
        self.last_actions = []
        self.action_sequence_length = 5
        self.position_history = []
        self.last_green_percent = 0
        self.green_percent = 0
        self.collected_food = 0

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
        speed = abs(left_motor_speed + right_motor_speed)

        return speed

    def calculate_reward(self, sensor_max=200):
        # Big reward for collecting food
        if self.rob.nr_food_collected() > self.collected_food:
            print("\n FOOD COLLECTED \n")
            self.collected_food = self.rob.nr_food_collected()
            return 10
        # Same position penalty
        x, y = self.rob.get_position().x, self.rob.get_position().y
        x, y = round(x, 1), round(y, 1)
        print("position: ", x, y)
        if (x, y) in self.position_history:
            pos_p = 0.5
        else:
            pos_p = 0
        self.position_history.append((x, y))
        # IR readings penalty
        highest_ir = max(self.ir_readings)
        if self.green_percent > 0:
            ir_p = 0
        elif highest_ir >= sensor_max:
            ir_p = 10
        else:
            ir_p = (highest_ir - min(self.ir_readings)) / (
                sensor_max - min(self.ir_readings)
            )
        # Reward for going forward if green_percent is high
        if self.green_percent > 0.6 and self.action == Direction.FRONT:
            forward_reward = 5
        elif self.green_percent < 0.6 and (
            self.action == Direction.LEFT or self.action == Direction.RIGHT
        ):
            forward_reward = 5
        else:
            forward_reward = 0
        # Reward for moving towards the green box
        if self.green_percent > self.last_green_percent:
            green_box_reward = 5
        else:
            green_box_reward = -1
        # Reward
        return (self.green_percent - ir_p) * pos_p + forward_reward + green_box_reward

    def process_image(self, image, save_image=False):
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
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        # Keep only the green channel
        green_channel = masked_image[:, :, 1]

        if save_image:
            image_name = f"image_{self.image_counter}.png"
            print(f"Processing image {image_name}")
            cv2.imwrite(os.path.join(image_run_dir, image_name), green_channel)
            self.image_counter += 1  # Increment the counter

        return green_channel

    def get_green_dist_from_center(self, image):
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
            return score

    def early_termination(self):
        if all(reward < 0 for reward in self.past_rewards[-20:]):
            print("Early termination due to low rewards")
            return True
        else:
            return False

    def step(self, action):
        speed = 70
        duration = 300
        # Map integer action to Direction instance
        action = Direction(action)
        if action == Direction.FRONT:  # forward
            self.rob.move_blocking(speed, speed, duration)
        # elif action == Direction.BACK:  # backward
        #     self.rob.move_blocking(-speed, -speed, duration)
        elif action == Direction.LEFT:  # left
            self.rob.move_blocking(-speed / 2, speed / 2, duration)
        elif action == Direction.RIGHT:  # right
            self.rob.move_blocking(speed / 2, -speed / 2, duration)
        else:
            raise ValueError(f"Invalid action {action}")

        self.action = action
        self.actions.append(action)

        if isinstance(self.rob, SimulationRobobo) and not self.rob.is_running():
            raise RuntimeError("Simulation is not running")
        self.ir_readings = np.array(self.rob.read_irs())
        self.ir_readings = self.ir_readings[self.mask]
        self.ir_readings = np.nan_to_num(self.ir_readings, posinf=1e10)
        # Update the wheel positions and duration
        wheel_position = self.rob.read_wheels()
        self.left_wheel_pos = wheel_position.wheel_pos_l
        self.right_wheel_pos = wheel_position.wheel_pos_r
        self.duration = duration

        image = self.rob.get_image_front()
        image = self.process_image(image, save_image=False)
        self.last_green_percent = self.green_percent
        # self.green_percent = self.get_green_percent(image[:, :, 1])
        self.green_percent = self.get_green_dist_from_center(image)
        print(f"Green percent: {self.green_percent}")

        # Update the observation with the new features and the image
        self.observation = {
            "vector": np.concatenate(
                [
                    self.ir_readings,
                    np.array(
                        [
                            self.left_wheel_pos,
                            self.right_wheel_pos,
                            self.green_percent,
                            self.last_green_percent,
                        ]
                    ),
                ]
            ),
            "image": image,
        }

        self.reward = self.calculate_reward()
        print("Reward: ", self.reward)
        self.past_rewards.append(self.reward)

        terminated = self.early_termination()
        truncated = False
        info = {}

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
        # Set phone pan and tilt
        # self.rob.set_phone_pan(50, 50)
        self.collected_food = 0
        self.rob.set_phone_tilt(109, 50)
        # Read IR
        self.ir_readings = np.array(self.rob.read_irs())
        self.ir_readings = self.ir_readings[self.mask]
        self.ir_readings = np.nan_to_num(self.ir_readings, posinf=1e10)
        # Update the wheel positions and duration
        wheel_position = self.rob.read_wheels()
        self.left_wheel_pos = wheel_position.wheel_pos_l
        self.right_wheel_pos = wheel_position.wheel_pos_r
        self.duration = 0  # Reset the duration to 0
        # Reset exploration grid
        self.grid = np.zeros(self.grid_size)
        # Get image
        image = self.rob.get_image_front()
        image = self.process_image(image, save_image=False)
        self.last_green_percent = self.green_percent
        self.green_percent = self.get_green_dist_from_center(image)
        # print(f"Green percent: {self.green_percent}")

        # Update the observation with the new features and the image
        self.observation = {
            "vector": np.concatenate(
                [
                    self.ir_readings,
                    np.array(
                        [
                            self.left_wheel_pos,
                            self.right_wheel_pos,
                            self.green_percent,
                            self.last_green_percent,
                        ]
                    ),
                ]
            ),
            "image": image,
        }
        info = {}

        return self.observation, info

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
        ir_readings = np.array(self.training_env.envs[0].ir_readings)
        left_motor_speed = np.array(self.training_env.envs[0].left_motor_speed)
        right_motor_speed = np.array(self.training_env.envs[0].right_motor_speed)

        for sensor_name, index in IR_SENSOR_INDICES.items():
            self.tb_formatter.writer.add_scalar(
                f"observations/{sensor_name}", ir_readings[index], self.num_timesteps
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
        # self._log_observations()
        self._log_rewards()
        self._log_action()
        self._log_episode_length()

        return True


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
        # check_env(env)
        return env

    vec_env = make_vec_env(
        make_env, n_envs=n_envs, monitor_dir=os.path.join(model_dir, "monitor")
    )
    DQN_PARAMS = dict(
        policy="MultiInputPolicy",
        env=vec_env,
        verbose=1,
        train_freq=50,
        gradient_steps=-1,
        gamma=0.99,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        target_update_interval=5,
        learning_starts=10,
        buffer_size=1000,
        batch_size=256,
        learning_rate=0.001,
        policy_kwargs=dict(
            net_arch=[256, 256],
            normalize_images=False,
        ),
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
        train_model(
            rob,
            n_episodes=50,
            time_steps_per_episode=100,
            verbose=1,
            load_model=False,
            n_envs=1,
            # model_name="DQN-20240614-020415_easy_50ep1kts",
        )
        # run_model(rob, model_name="DQN-20240619-232040", n_steps=500, verbose=1)
    elif isinstance(rob, HardwareRobobo):
        test_hardware(rob, "DQN-20240619-232040", n_steps=200)
