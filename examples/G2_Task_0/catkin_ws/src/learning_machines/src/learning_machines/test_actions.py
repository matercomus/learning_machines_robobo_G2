import cv2

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


def move_until_obstacle(
    rob: HardwareRobobo,
    speed: int,
    direction: str,
    duration: int = 250,
):
    """Move the robot in the specified direction until an obstacle is detected
    [BackL, BackR, FrontL, FrontR, FrontC, FrontRR, BackC, FrontLL]
    """
    ir_dist = 50
    while True:
        irs = rob.read_irs()
        if irs[2] >= ir_dist or irs[3] >= ir_dist or irs[4] >= ir_dist:
            print("Obstacle detected in front")
            print(irs)
            break
        if irs[0] >= ir_dist or irs[1] >= ir_dist or irs[6] >= ir_dist:
            print("Obstacle detected in the back")
            print(irs)
            break
        if irs[7] >= ir_dist or irs[2] >= ir_dist:
            print("Obstacle detected on the left")
            print(irs)
            break
        if irs[3] >= ir_dist or irs[5] >= ir_dist:
            print("Obstacle detected on the right")
            print(irs)
            break
        if direction == "forward":
            rob.move_blocking(speed, speed, duration)
        if direction == "backward":
            rob.move_blocking(-speed, -speed, duration)

    rob.move_blocking(0, 0, 100)  # Stop the robot


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


def test_hardware(rob: HardwareRobobo):
    print("Phone battery level: ", rob.read_phone_battery())
    print("Robot battery level: ", rob.read_robot_battery())
    print("IRS data: ", rob.read_irs())

    speed = 50
    move_until_obstacle(rob, speed, "forward")
    rob.move_blocking(-speed, -speed, 500)  # Move back a bit
    rob.move_blocking(speed, -speed, 500)  # Turn right
    rob.move_blocking(0, 0, 100)  # Stop the robot
    move_until_obstacle(rob, speed, "forward")


def run_all_actions(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    if isinstance(rob, SimulationRobobo):
        test_sim(rob)
    if isinstance(rob, HardwareRobobo):
        test_hardware(rob)
    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
