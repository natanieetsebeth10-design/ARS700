"""
COMPONENT A step 1: Sensor simulation and poerceotion

This script controls the e-puck robot in Webots.

What the code does:
1. Reads data from the 8 IR distance sensors.
2. Takes images using the robot's camera.
3. Uses OpenCV to detect coloured obstacles in the images.
4. Saves the sensor readings and detected obstacles to a CSV file.
5. Keeps the robot stationary while testing the sensors.

The saved data can be used later to create maps and visualise obstacles.

Save this file as epuck_perception.py inside the robot's controller folder.
"""

from controller import Robot
import numpy as np
import cv2
import csv
import os


# Set up the robot and simulation time step.
robot = Robot()
TIME_STEP = int(robot.getBasicTimeStep())


# Enable the 8 IR sensors so the robot can detect nearby obstacles.
NUM_SENSORS = 8
distance_sensors = []

for i in range(NUM_SENSORS):
    sensor = robot.getDevice(f"ps{i}")
    sensor.enable(TIME_STEP)
    distance_sensors.append(sensor)


# Turn on the camera and get the image size.
camera = robot.getDevice("camera")
camera.enable(TIME_STEP)

cam_width = camera.getWidth()
cam_height = camera.getHeight()


# Set up the wheel motors.
# The robot stays still while we test the sensors and camera.
left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)


# Create a CSV file to save the sensor readings and camera results.
os.makedirs("../../data", exist_ok=True)

log_path = "../../data/perception_log.csv"
log_file = open(log_path, mode="w", newline="")

log_writer = csv.writer(log_file)

# Add column names to the CSV file.
log_writer.writerow(
    ["timestep"] + [f"ps{i}" for i in range(NUM_SENSORS)]
    + ["num_obstacles_detected", "obstacle_centroids_xy"]
)


def detect_obstacles(image_bgr):
    """
    Detect coloured obstacles in the camera image.

    1. Convert the image to HSV colour format.
    2. Find coloured areas and separate them from the background.
    3. Clean up small gaps and noise in the image.
    4. Find the outlines of the obstacles.
    5. Calculate the centre of each detected obstacle.
    """

    # Convert the image to HSV to make colour detection easier.
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Find coloured areas and ignore most grey or white background areas.
    lower = np.array([0, 80, 40])
    upper = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    # Clean up small gaps in the detected areas.
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find the outlines of the detected objects.
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    centroids = []

    # Find the centre of each large enough object.
    for c in contours:
        area = cv2.contourArea(c)

        # Ignore very small areas that are probably noise.
        if area < 15:
            continue

        M = cv2.moments(c)

        # Skip the object if its centre cannot be calculated.
        if M["m00"] == 0:
            continue

        # Calculate the centre point of the detected object.
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        centroids.append((cx, cy))

    return centroids


# Start the main loop and count the simulation steps.
step_count = 0
MAX_STEPS = 300  # Stop after 300 steps.


while robot.step(TIME_STEP) != -1 and step_count < MAX_STEPS:

    # Read the current values from all 8 IR sensors.
    ps_values = [s.getValue() for s in distance_sensors]

    # Get the image from the camera.
    image = camera.getImage()

    # Convert the image data into an array that OpenCV can process.
    img_array = np.frombuffer(image, np.uint8).reshape(
        (cam_height, cam_width, 4)
    )

    # Remove the alpha channel to get the image's colour data.
    img_bgr = img_array[:, :, :3]

    # Detect obstacles and find their centre points.
    centroids = detect_obstacles(img_bgr)

    # Save the sensor readings and detected obstacles to the CSV file.
    log_writer.writerow(
        [step_count] + ps_values + [len(centroids), centroids]
    )

    # Keep both wheels stopped while testing the perception system.
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    step_count += 1


# Close the file after collecting the data.
log_file.close()

print(
    f"Perception logging complete. {step_count} steps written to {log_path}"
)