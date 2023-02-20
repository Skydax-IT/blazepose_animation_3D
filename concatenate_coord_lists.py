import utils

# Frames path
frame_angle1 = [r"full path side frame path"] # Side frame
frame_angle2 = [r"full path face frame path"] # Face frame

# Create a list of 3D coordinates from BlazePose for two different angles
coordinates_angle1 = utils.compute_coordinates(frame_angle1)
coordinates_angle2 = utils.compute_coordinates(frame_angle2)

# Print
print ("\nAngle 1: \n", coordinates_angle1)
print ("\nAngle 2:\n", coordinates_angle2)


# Assume you have a list of frames, where each frame is a tuple of 3D coordinates for each angle
frames = [(coordinates_angle1, coordinates_angle2)]

# Process each frame to generate 3D coordinates for landmarks from two different angles
landmarks = []
for frame in frames:
    landmarks.append(utils.merge_coordinates(*frame))

print ("\n3D Landmarks:\n", landmarks)
# utils.plot_landmarks(landmarks)







