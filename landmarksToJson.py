import numpy as np
import json
import sys

def npzToJson(filename):
    data = np.load(filename)
    keypoints_array = data['keypoints']
    
    # Convert to a JSON-serializable format
    json_data = {}

    for i, frame in enumerate(keypoints_array):
        if np.isnan(frame).all():
            json_data[f"frame_{i}"] = None  # no face detected
        else:
            json_data[f"frame_{i}"] = frame.tolist()

    # Save to JSON
    output_filename = filename.replace('.npz', '.json')
    with open(output_filename, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"Saved landmarks to {output_filename}")
    
if __name__ == "__main__":
    npz_file = sys.argv[1]
    npzToJson(npz_file)
