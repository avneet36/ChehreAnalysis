import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Facial features mapping 
FACIAL_FEATURES = {
    "jawline": list(range(0, 17)),
    "right_eyebrow": list(range(17, 22)),
    "left_eyebrow": list(range(22, 27)),
    "nose": list(range(27, 36)),
    "right_eye": list(range(36, 42)),
    "left_eye": list(range(42, 48)),
    "outer_lip": list(range(48, 60)),
    "inner_lip": list(range(60, 68)),
}

# Colors for different facial features (RGB normalized to 0-1 for matplotlib)
FEATURE_COLORS = {
    "jawline": (1.0, 0.0, 0.0),      # Red
    "right_eyebrow": (0.0, 1.0, 0.0),# Green
    "left_eyebrow": (0.0, 0.0, 1.0), # Blue
    "nose": (1.0, 1.0, 0.0),         # Yellow
    "right_eye": (1.0, 0.0, 1.0),    # Magenta
    "left_eye": (0.0, 1.0, 1.0),     # Cyan
    "outer_lip": (0.5, 0.0, 0.5),    # Purple
    "inner_lip": (0.0, 0.5, 0.5),    # Teal
}

def load_keypoints(npz_file_path):
    try:
        data = np.load(npz_file_path)
        keypoints = data['keypoints']
        print(f"Loaded keypoints shape: {keypoints.shape}")
        return keypoints
    except Exception as e:
        print(f"Error loading {npz_file_path}: {e}")
        return None

def visualize_frame(keypoints_frame, frame_idx, save_path=None, show_connections=True):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Set black background and 256x256 limits
    ax.set_facecolor('black')
    ax.set_xlim(0, 1) 
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Invert y-axis to match image coordinates
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Check if keypoints are valid (not all NaN)
    if np.all(np.isnan(keypoints_frame)):
        ax.text(0.5, 0.5, 'No Face Detected', ha='center', va='center', 
                color='white', fontsize=16)
        ax.set_title(f'Frame {frame_idx}: No Face Detected', color='white')
        if save_path:
            plt.savefig(save_path, facecolor='black', bbox_inches='tight', dpi=100)
        return fig, ax
    
    # Plot each facial feature with different colors
    for feature_name, indices in FACIAL_FEATURES.items():
        points = keypoints_frame[indices]
        
        # Skip if points are NaN
        if np.any(np.isnan(points)):
            continue
            
        color = FEATURE_COLORS[feature_name]
        
        # Plot points
        ax.scatter(points[:, 0], points[:, 1], 
                  c=[color], s=20, alpha=0.8, edgecolors='white', linewidth=0.5)
        
        # Draw connections for certain features
        # if show_connections:
        #     if feature_name in ['jawline', 'right_eyebrow', 'left_eyebrow', 'nose']:
        #         # Draw lines connecting consecutive points
        #         for i in range(len(points) - 1):
        #             ax.plot([points[i, 0], points[i+1, 0]], 
        #                    [points[i, 1], points[i+1, 1]], 
        #                    color=color, alpha=0.6, linewidth=1)
            
        #     elif feature_name in ['right_eye', 'left_eye', 'outer_lip', 'inner_lip']:
        #         # Draw closed loops for eyes and lips
        #         for i in range(len(points)):
        #             next_i = (i + 1) % len(points)
        #             ax.plot([points[i, 0], points[next_i, 0]], 
        #                    [points[i, 1], points[next_i, 1]], 
        #                    color=color, alpha=0.6, linewidth=1)
    
    ax.set_title(f'Frame {frame_idx}: Facial Keypoints', color='white', fontsize=14)
    
    if save_path:
        plt.savefig(save_path, facecolor='black', bbox_inches='tight', dpi=100)
    
    return fig, ax

def visualize_video_keypoints(npz_file_path, output_dir=None, max_frames=None, 
                             show_connections=True, save_individual=False):
    keypoints = load_keypoints(npz_file_path)
    if keypoints is None:
        return
    
    video_name = Path(npz_file_path).stem.replace('_landmarks', '')
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    num_frames = keypoints.shape[0]
    frames_to_process = min(num_frames, max_frames) if max_frames else num_frames
    
    print(f"Visualizing {frames_to_process} frames from {video_name}")
    
    # Create a grid visualization for overview
    cols = min(8, frames_to_process)
    rows = (frames_to_process - 1) // cols + 1
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    if frames_to_process == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i in range(frames_to_process):
        if rows > 1 or cols > 1:
            ax = axes[i]
        else:
            ax = axes[0]
        
        keypoints_frame = keypoints[i]
        
        # Set up the subplot
        ax.set_facecolor('black')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        
        if not np.all(np.isnan(keypoints_frame)):
            for feature_name, indices in FACIAL_FEATURES.items():
                points = keypoints_frame[indices]
                if np.any(np.isnan(points)):
                    continue
                color = FEATURE_COLORS[feature_name]
                ax.scatter(points[:, 0], points[:, 1], 
                          c=[color], s=5, alpha=0.8)
        
        ax.set_title(f'F{i}', color='white', fontsize=8)
        
        # Save individual frame if requested
        if save_individual and output_dir:
            individual_fig, individual_ax = visualize_frame(
                keypoints_frame, i, 
                save_path=os.path.join(output_dir, f'{video_name}_frame_{i:04d}.png'),
                show_connections=show_connections
            )
            plt.close(individual_fig)
    
    # Hide unused subplots
    for i in range(frames_to_process, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Keypoints Overview: {video_name}', color='white', fontsize=16)
    plt.tight_layout()
    
    if output_dir:
        overview_path = os.path.join(output_dir, f'{video_name}_overview.png')
        plt.savefig(overview_path, facecolor='black', bbox_inches='tight', dpi=150)
        print(f"Overview saved to: {overview_path}")
    
    plt.show()

def main():
    npz_file_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(npz_file_path):
        print(f"Error: File {npz_file_path} not found")
        return
    
    print(f"Visualizing keypoints from: {npz_file_path}")
    
    # Create overview visualization
    visualize_video_keypoints(npz_file_path, output_dir, max_frames=50, 
                             show_connections=True, save_individual=True)
if __name__ == "__main__":
    main()