import face_alignment
from face_alignment import LandmarksType
import cv2
import numpy as np
from tqdm import tqdm

'''
What Are the 68 Points?
They follow a standard facial landmark mapping, like:

Index	Facial Feature
0–16	Jawline
17–21	Right eyebrow
22–26	Left eyebrow
27–35	Nose bridge + tip
36–41	Right eye
42–47	Left eye
48–59	Outer lip
60–67	Inner lip

'''

# Load the face alignment model (2D)
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cpu')

# Path to your video
cap = cv2.VideoCapture("video1.mp4")

#for debugging: confirm openCV is loading the video
print("Loaded?", cap.isOpened())

#Split video into frames & get number of frames
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Frame count:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

# Store keypoints here
all_keypoints = []

# Read and process each frame
for _ in tqdm(range(frame_count), desc="Processing frames"):
    ret, frame = cap.read()
    if not ret:
        break
    preds = fa.get_landmarks(frame)
    if preds is not None:
        all_keypoints.append(preds[0])  # Only take first face
    else:
        all_keypoints.append(np.full((68, 2), np.nan))  # fill with NaNs if no face

cap.release()

# Convert to numpy and save
keypoints_array = np.array(all_keypoints)
np.save("landmarks_2d.npy", keypoints_array)

print("Saved 2D landmarks to landmarks_2d.npy")
