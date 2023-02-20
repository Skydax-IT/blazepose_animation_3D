# blazepose_animation_3D
Project that aims to animate a 3D model from coordinates retrieved with BlazePose. Cancelled due to lack of time.

## conconcatenate_coord_lists.py explanation
The function 'concatenate_coord_lists' combines two lists of coordinates extracted from a frame captured at different angles. It selects the X coordinate from the first list and the Y and Z coordinates from the second list, merging them into a single list of three-dimensional coordinates.

## sync_videos_landmarks.ipynb explanation
The 'sync_videos_landmarks' function reads two videos captured from different angles, and extracts the coordinates of the four corners of a rectangle representing the face in each frame of both videos using the dlib frontal face detector.

## Author 
Enzo "Skydax-IT" Licata

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
