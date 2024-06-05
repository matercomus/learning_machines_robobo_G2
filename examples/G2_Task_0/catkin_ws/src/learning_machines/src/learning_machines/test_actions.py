import cv2

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


# Mapping of directions to their corresponding IR sensor indices
DIRECTION_IR_MAP = {
    Direction.FRONT: [2, 3, 4],  # FrontL, FrontR, FrontC
    Direction.BACK: [0, 1, 6],  # BackL, BackR, BackC
    Direction.LEFT: [7, 2],  # FrontLL, FrontL
    Direction.RIGHT: [3, 5],  # FrontR, FrontRR
}


def detect_obstacle(
    irs: List[Optional[float]], ir_dist: int, dampening_factor: float
) -> List[Tuple[Direction, float]]:
    """Detect the direction of the obstacle and return all detected directions
    sorted by proximity."""
    detected_directions = []

    for direction, indices in DIRECTION_IR_MAP.items():
        adjusted_ir_values = [
            (
                irs[idx] * (dampening_factor if idx != indices[0] else 1)
                if idx < len(irs) and irs[idx] is not None
                else 0
            )
            for idx in indices
        ]
        max_adjusted_ir_value = max(adjusted_ir_values)

        if max_adjusted_ir_value >= ir_dist:
            detected_directions.append((direction, max_adjusted_ir_value))

    # Sort directions by the IR value in descending order (closest obstacle first)
    detected_directions.sort(key=lambda x: x[1], reverse=True)

    if detected_directions:
        print(
            f"Obstacles detected in {[direction.value for direction, _ in detected_directions]}"
        )
        print("IRS data: ", irs)

    return detected_directions


def move_until_obstacle(
    rob: "HardwareRobobo",
    speed: int,
    direction: str,
    duration: int = 200,
    dampening_factor: float = 0.9,
) -> List[Tuple[Direction, float]]:
    """Move the robot in the specified direction until an obstacle is detected"""
    ir_dist = 50
    while True:
        irs = rob.read_irs()
        detected_directions = detect_obstacle(irs, ir_dist, dampening_factor)
        if detected_directions:
            break
        if direction == "forward":
            print("Moving forward")
            rob.move_blocking(speed, speed, duration)
        if direction == "backward":
            print("Moving backward")
            rob.move_blocking(-speed, -speed, duration)

    rob.move_blocking(0, 0, 100)  # Stop the robot
    return detected_directions


def move_away_from_obstacle(
    rob: "HardwareRobobo",
    obstacle_directions: List[Tuple[Direction, float]],
    speed: int,
):
    """Move the robot in the opposite direction of the detected obstacles"""
    for obstacle_direction, _ in obstacle_directions:
        if obstacle_direction == Direction.FRONT:
            print("Moving backward")
            rob.move_blocking(-speed, -speed, 1000)
            print("Turning right")
            rob.move_blocking(speed, -speed, 500)
        elif obstacle_direction == Direction.BACK:
            print("Moving forward")
            rob.move_blocking(speed, speed, 1000)
            print("Turning left")
            rob.move_blocking(-speed, speed, 500)
        elif obstacle_direction == Direction.LEFT:
            print("Turning right")
            rob.move_blocking(speed, -speed, 500)
        elif obstacle_direction == Direction.RIGHT:
            print("Turning left")
            rob.move_blocking(-speed, speed, 500)
    rob.move_blocking(0, 0, 100)  # Stop the robot


def test_hardware(rob: "HardwareRobobo"):
    print("Phone battery level: ", rob.read_phone_battery())
    print("Robot battery level: ", rob.read_robot_battery())
    print("IRS data: ", rob.read_irs())
    speed = 50
    n_turns = 10

    while n_turns > 0:
        obstacle_directions = move_until_obstacle(rob, speed, "forward")
        move_away_from_obstacle(rob, obstacle_directions, speed)
        n_turns -= 1


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
