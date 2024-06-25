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
lower_red = np.array([160, 155, 84])
upper_red = np.array([179, 255, 255])

lower_red3 = np.array([0, 50, 20])
upper_red3 = np.array([5, 255, 255])

lower_red4 = np.array([175, 50, 20])
upper_red4 = np.array([180, 255, 255])


def process_image(
    image, color_lower1, color_upper1, color_lower2=None, color_upper2=None
):
    # Resize the image to 64x64 pixels
    image = cv2.resize(image, (64, 64))
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Mask the image
    mask1 = cv2.inRange(hsv_image, color_lower1, color_upper1)
    if color_lower2 is not None and color_upper2 is not None:
        mask2 = cv2.inRange(hsv_image, color_lower2, color_upper2)
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        mask = mask1
    return cv2.bitwise_and(image, image, mask=mask)


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
        processed_image_red = process_image(image, lower_red, upper_red)

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
