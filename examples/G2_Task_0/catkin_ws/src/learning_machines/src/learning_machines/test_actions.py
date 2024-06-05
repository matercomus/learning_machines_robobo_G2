import cv2
import time

from enum import Enum
from typing import List, Optional, Tuple
from data_files import FIGRURES_DIR
from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
    HardwareRobobo,
)


def test_emotions(rob: IRobobo):
    rob.set_emotion(Emotion.HAPPY)
    rob.talk("Hello")
    rob.play_emotion_sound(SoundEmotion.PURR)
    rob.set_led(LedId.FRONTCENTER, LedColor.GREEN)


def test_move_and_wheel_reset(rob: IRobobo):
    rob.move_blocking(100, 100, 1000)
    print("before reset: ", rob.read_wheels())
    rob.reset_wheels()
    rob.sleep(1)
    print("after reset: ", rob.read_wheels())


def test_sensors(rob: IRobobo):
    print("IRS data: ", rob.read_irs())
    image = rob.get_image_front()
    cv2.imwrite(str(FIGRURES_DIR / "photo.png"), image)
    print("Phone pan: ", rob.read_phone_pan())
    print("Phone tilt: ", rob.read_phone_tilt())
    print("Current acceleration: ", rob.read_accel())
    print("Current orientation: ", rob.read_orientation())


def test_phone_movement(rob: IRobobo):
    rob.set_phone_pan_blocking(20, 100)
    print("Phone pan after move to 20: ", rob.read_phone_pan())
    rob.set_phone_tilt_blocking(50, 100)
    print("Phone tilt after move to 50: ", rob.read_phone_tilt())


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


def turn_away_from_obstacle(
    rob: "HardwareRobobo",
    speed: int = 50,
    move_duration: int = 300,
    turn_duration: int = 100,
    ir_threshold: int = 50,
):
    """Turns the robot away from detected obstacles until all front sensors are
    clear."""
    while True:
        irs = rob.read_irs()

        # If any front sensor detects an obstacle, move backward once and then
        # keep turning
        front_sensors = [
            irs[IR_SENSOR_INDICES[sensor_name]]
            for sensor_name in DIRECTION_IR_MAP[Direction.FRONT]
        ]
        if any(ir >= ir_threshold for ir in front_sensors):
            print("Obstacle detected, moving backward")
            rob.move_blocking(-speed, -speed, move_duration)
            print("Turning right")
            while any(ir >= ir_threshold for ir in front_sensors):
                rob.move_blocking(speed, -speed, turn_duration)
                irs = rob.read_irs()
                front_sensors = [
                    irs[IR_SENSOR_INDICES[sensor_name]]
                    for sensor_name in DIRECTION_IR_MAP[Direction.FRONT]
                ]
        else:
            print("Path is clear, moving forward")
            break

    rob.move_blocking(speed, speed, move_duration)  # Move forward


def test_hardware(rob: "HardwareRobobo", n_seconds=120):
    print("Phone battery level: ", rob.read_phone_battery())
    print("Robot battery level: ", rob.read_robot_battery())
    print("IRS data: ", rob.read_irs())
    speed = 50

    start_time = time.time()
    while (time.time() - start_time) < n_seconds:
        turn_away_from_obstacle(rob, speed)


# Actions to perform in simulation
def test_sim(rob: SimulationRobobo):
    print(rob.get_sim_time())
    print(rob.is_running())
    print(rob.get_position())
    print("IRS data: ", rob.read_irs())

    ir_dist = 200
    while (
        rob.read_irs()[2] < ir_dist
        and rob.read_irs()[4] < ir_dist
        and rob.read_irs()[3] < ir_dist
    ):
        rob.move_blocking(25, 25, 500)
    print("Reached obstacle")
    # rob.move_blocking(0, 0, 100)
    print("IRS data: ", rob.read_irs())
    rob.move_blocking(-50, -50, 500)
    rob.move_blocking(0, 0, 10)
    rob.move_blocking(50, 0, 1000)
    rob.move_blocking(50, 50, 3000)


def run_all_actions(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    if isinstance(rob, SimulationRobobo):
        test_sim(rob)
    if isinstance(rob, HardwareRobobo):
        test_hardware(rob)
    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
