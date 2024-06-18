import cv2
import os
import numpy as np

calib_photos_dir = "/home/matt/Dev/MS_AI/learning_machines_robobo_G2/examples/G2_Task_2/LM_Nokia_color_calib_photos/"
image_run_dir = "/home/matt/Dev/MS_AI/learning_machines_robobo_G2/examples/G2_Task_2/LM_Nokia_color_calib_photos_processed_stiched/"
os.makedirs(image_run_dir, exist_ok=True)


image_counter = 0


def process_image(image, save_image=False):
    global image_counter
    image_name = f"image_{image_counter}.png"
    print(f"Processing image {image_name}")
    # Resize the image to 64x64 pixels
    image = cv2.resize(image, (64, 64))
    # Flip the image back
    # image = cv2.flip(image, 0)
    # Isolate green channel
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    # Mask the image
    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Stitch the original image with the processed image
    stitched_image = cv2.hconcat([image, masked_image])

    if save_image:
        cv2.imwrite(
            os.path.join(image_run_dir, image_name),
            stitched_image,
        )
        image_counter += 1  # Increment the counter

    return stitched_image


for image in os.listdir(calib_photos_dir):
    image_path = os.path.join(calib_photos_dir, image)
    image = cv2.imread(image_path)
    process_image(image, save_image=True)
