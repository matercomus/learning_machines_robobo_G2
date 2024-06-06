import cv2
import time
import pandas as pd
import numpy as np
import os

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


def check_obstacles(
    rob: "HardwareRobobo", direction_map: dict, sensor_indices: dict, ir_threshold: int
):
    """Check for obstacles in the specified directions."""
    irs = rob.read_irs()
    sensors = {
        direction: [
            irs[sensor_indices[sensor_name]] for sensor_name in direction_map[direction]
        ]
        for direction in direction_map
    }
    obstacles = {
        direction: any(ir >= ir_threshold for ir in sensors[direction])
        for direction in sensors
    }
    return obstacles


def turn_away_from_obstacle(
    rob: "HardwareRobobo",
    speed: int = 50,
    move_duration: int = 200,
    turn_duration: int = 100,
    ir_threshold: int = 50,
):
    """Turns the robot away from detected obstacles until all front sensors are
    clear or obstacle detected behind."""
    speed_normal = speed
    speed_turn = speed // 3

    while True:
        obstacles = check_obstacles(
            rob, DIRECTION_IR_MAP, IR_SENSOR_INDICES, ir_threshold
        )
        obstacle_ahead = obstacles[Direction.FRONT]
        obstacle_behind = obstacles[Direction.BACK]

        if obstacle_ahead:
            print("Obstacle detected ahead, moving backward")
            rob.move_blocking(-speed_normal, -speed_normal, move_duration)
            while obstacle_ahead:
                print("Turning right")
                rob.move_blocking(speed_turn, -speed_turn, turn_duration)
                obstacles = check_obstacles(
                    rob, DIRECTION_IR_MAP, IR_SENSOR_INDICES, ir_threshold
                )
                obstacle_ahead = obstacles[Direction.FRONT]
                obstacle_left = obstacles[Direction.LEFT]
                obstacle_behind = obstacles[Direction.BACK]

                if not obstacle_ahead or obstacle_left or obstacle_behind:
                    break
        else:
            break


def test_hardware2(rob: "HardwareRobobo", n_seconds=120):
    print("Phone battery level: ", rob.read_phone_battery())
    print("Robot battery level: ", rob.read_robot_battery())
    print("IRS data: ", rob.read_irs())
    speed = 50
    start_time = time.time()

    while (time.time() - start_time) < n_seconds:
        turn_away_from_obstacle(rob, speed)
        print("Moving forward")
        rob.move_blocking(speed, speed, 200)  # Move forward


# Actions to perform in simulation
def test_sim_old(rob: SimulationRobobo):
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


def time_now(start_time):
    time_now = time.time() - start_time
    time_now = round(time_now, 3)
    return time_now


def write_data(data, time, irs, direction=None, event=None, run_nr=0):
    data["time"].append(time_now(time))
    for sensor_name, sensor_index in IR_SENSOR_INDICES.items():
        data[sensor_name].append(irs[sensor_index])
    data["direction"].append(direction if direction is not None else "N/A")
    data["event"].append(event if event is not None else "N/A")
    data["run_nr"].append(run_nr)


def test_hardware(rob: "HardwareRobobo", mode="HW"):
    test(rob, mode=mode, ir_threshold=50)


def test_sim(rob: "SimulationRobobo", mode="SIM"):
    data = dict()
    for sensor_name in IR_SENSOR_INDICES.keys():
        data[sensor_name] = []

    df = None

    for i in range(11):
        x = test(rob, run_nr=i, data=data, mode=mode, ir_threshold=200)
        if i > 0:  # Start saving the results after the first run
            if df is None:
                df = pd.DataFrame(x)
            else:
                df = pd.concat([df, pd.DataFrame(x)], ignore_index=True)
        rob.stop_simulation()
        rob.play_simulation()

    if df is not None:
        os.makedirs("/root/results/data", exist_ok=True)
        df.to_csv(f"/root/results/data/{mode}_run_final.csv", index=False)


def test(
    rob,
    run_nr,
    mode,
    data,
    ir_threshold,
    speed: int = 50,
    move_duration: int = 200,
    turn_duration: int = 230,
):
    print("Running test in ", mode, " mode")
    start_time = time.time()
    write_data(
        data=data,
        time=start_time,
        irs=rob.read_irs(),
        direction="forward",
        run_nr=run_nr,
    )

    while (
        (
            rob.read_irs()[IR_SENSOR_INDICES["FrontL"]] < ir_threshold
            and rob.read_irs()[IR_SENSOR_INDICES["FrontL"]] != float("inf")
        )
        and (
            rob.read_irs()[IR_SENSOR_INDICES["FrontC"]] < ir_threshold
            and rob.read_irs()[IR_SENSOR_INDICES["FrontC"]] != float("inf")
        )
        and (
            rob.read_irs()[IR_SENSOR_INDICES["FrontR"]] < ir_threshold
            and rob.read_irs()[IR_SENSOR_INDICES["FrontR"]] != float("inf")
        )
    ):
        write_data(
            data=data,
            time=start_time,
            irs=rob.read_irs(),
            direction="forward",
            run_nr=run_nr,
        )
        rob.move_blocking(speed, speed, move_duration)

    write_data(
        data=data,
        time=start_time,
        irs=rob.read_irs(),
        direction="forward",
        event="obstacle",
        run_nr=run_nr,
    )

    for _ in range(3):
        rob.move_blocking(-speed, -speed, move_duration)
        write_data(
            data=data,
            time=start_time,
            irs=rob.read_irs(),
            direction="backward",
            run_nr=run_nr,
        )

    for _ in range(5):
        rob.move_blocking(speed, -speed, turn_duration)
        write_data(
            data=data,
            time=start_time,
            irs=rob.read_irs(),
            direction="right",
            run_nr=run_nr,
        )

    for _ in range(10):
        rob.move_blocking(speed, speed, move_duration)
        write_data(
            data=data,
            time=start_time,
            irs=rob.read_irs(),
            direction="forward",
            run_nr=run_nr,
        )

    del ir_threshold
    return data


def run_all_actions(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    if isinstance(rob, SimulationRobobo):
        test_sim(rob)
    if isinstance(rob, HardwareRobobo):
        test_hardware(rob)
    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
