import os
import cv2
import numpy as np

images_dir = "/home/matt/Downloads/Photos-task3"
output_dir = "/home/matt/Downloads/Photos-task3-output"
os.makedirs(output_dir, exist_ok=True)

lower_green = np.array([40, 40, 40])
upper_green = np.array([80, 255, 255])

lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])
# (hMin = 0 , sMin = 0, vMin = 0), (hMax = 92 , sMax = 131, vMax = 255)
lower_red = np.array([0, 0, 0])
upper_red = np.array([92, 131, 255])


def process_image(
    image, color_lower1, color_upper1, color_lower2=None, color_upper2=None
):
    # Resize the image to 64x64 pixels
    image = cv2.resize(image, (64, 64))
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # Mask the image
    mask1 = cv2.inRange(hsv_image, color_lower1, color_upper1)
    if color_lower2 is not None and color_upper2 is not None:
        mask2 = cv2.inRange(hsv_image, color_lower2, color_upper2)
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        mask = mask1
    return cv2.bitwise_and(image, image, mask=mask)


def process_cyan(image):
    image = cv2.resize(image, (64, 64))
    # Convert the RGB image to BGR
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Invert the BGR image
    bgr_inv = cv2.bitwise_not(bgr)

    # Convert the inverted image to HSV
    hsv_inv = cv2.cvtColor(bgr_inv, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the cyan color in HSV
    lower_cyan = np.array([80, 70, 50])
    upper_cyan = np.array([100, 255, 255])

    # Create a mask for the cyan color
    mask = cv2.inRange(hsv_inv, lower_cyan, upper_cyan)

    return cv2.bitwise_and(hsv_inv, hsv_inv, mask=mask)


def get_color_cell_and_percent(image):
    # Define the grid size
    grid_size = 3
    # Get the size of each cell
    cell_size = image.shape[0] // grid_size
    # Initialize the list to store the percentage of non-black pixels in each cell
    green_percent = []
    # Loop over the grid cells
    for i in range(grid_size):
        for j in range(grid_size):
            # Get the current cell
            cell = image[
                i * cell_size : (i + 1) * cell_size,
                j * cell_size : (j + 1) * cell_size,
            ]
            # Calculate the percentage of non-black pixels in the current cell
            non_black_pixels = np.sum(cell != 0)
            total_pixels = cell_size * cell_size
            green_percent.append(
                (i * grid_size + j, round(non_black_pixels / total_pixels, 3))
            )
    return green_percent


def main():
    # Get the list of image files
    image_files = os.listdir(images_dir)
    image_id = 0
    for image_file in image_files:
        # Load the image
        image = cv2.imread(os.path.join(images_dir, image_file))
        # Process the image
        processed_image_green = process_image(image, lower_green, upper_green)
        # processed_image_red = process_image(
        #     image, lower_red1, upper_red1, lower_red2, upper_red2
        # )
        # processed_image_red = process_image(image, lower_red, upper_red)
        processed_image_red = process_cyan(image)
        # Get the green percentage
        green_percent = get_color_cell_and_percent(processed_image_green)
        red_percent = get_color_cell_and_percent(processed_image_red)
        # Stitch the original and processed images horizontally
        stitched_image = np.hstack(
            (cv2.resize(image, (64, 64)), processed_image_green, processed_image_red)
        )
        # Generate the filename for the processed image
        filename = f"{image_file}_green_percent_{image_id}.jpg"
        print("-" * 50)
        print(f"Saving {filename}")
        print(f"Green: {green_percent}")
        print(f"Red: {red_percent}")
        # Save the stitched image
        cv2.imwrite(os.path.join(output_dir, filename), stitched_image)
        image_id += 1


if __name__ == "__main__":
    main()
