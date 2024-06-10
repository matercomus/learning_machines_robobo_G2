import time
import datetime
import pandas as pd
import numpy as np
import os

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


def run_all_actions(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        test_simulation(rob)
    elif isinstance(rob, HardwareRobobo):
        test_hardware(rob)
