import numpy as np
import json

# Load the .npy data
data = np.load("landmarks_2d.npy")

# Convert to a JSON-serializable format
json_data = {}

for i, frame in enumerate(data):
    if np.isnan(frame).all():
        json_data[f"frame_{i}"] = None  # no face detected
    else:
        json_data[f"frame_{i}"] = frame.tolist()

# Save to JSON
with open("landmarks_2d.json", "w") as f:
    json.dump(json_data, f, indent=2)

print("Saved landmarks to landmarks_2d.json")
