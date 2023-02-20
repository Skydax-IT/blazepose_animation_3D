# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import tempfile

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


# Function that takes two sets of 3D coordinates and returns merged coordinates
def merge_coordinates(coordinates1, coordinates2):
    merged_coordinates = []
    for i in range(len(coordinates1)):
        x = coordinates1[i]['x']
        # y = (coordinates1[i]['y'] + coordinates2[i]['y'])/2
        y = coordinates2[i]['y']
        z = coordinates2[i]['z']
        merged_coordinates.append({'x': x, 'y': y, 'z': z})
    return merged_coordinates


# Function that compute the 3D coordinates representing the landmarks
def compute_coordinates(frame):
    # For static images:
    num = '8'
    BG_COLOR = (192, 192, 192) # gray
    coordinates = []

    with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True, min_detection_confidence=0.5) as pose:
        # create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg') as temp:
            temp_file = temp.name
            # write the frame to the temporary file
            cv2.imwrite(temp_file, frame)
            # read the image from the temporary file
            image = cv2.imread(temp_file)
            image_height, image_width, _ = image.shape
            # Convert the BGR image to RGB before processing.
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # if not results.pose_landmarks:
            #     break

            annotated_image = image.copy()
            # Draw segmentation on the image.
            # To improve segmentation around boundaries, consider applying a joint bilateral filter to "results.segmentation_mask" with "image".
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
            annotated_image = np.where(condition, annotated_image, bg_image)
            # print(">>>>>>>", results.pose_world_landmarks)
            for landmark in results.pose_world_landmarks.landmark:
                coordinates.append({'x': landmark.x, 'y': landmark.y, 'z': landmark.z, 'visibility': landmark.visibility})        

    return coordinates


# Function that plot the landmark in a 3D graph
def plot_landmarks(landmarks):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the landmarks
    for coord in landmarks[0]:
        x = coord['x']
        y = coord['y']
        z = coord['z']
        ax.scatter(x, y, z, c='r', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


        
