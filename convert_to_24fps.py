import subprocess
import os
import json
import shutil

# Detects actual FPS of given video
def get_fps(video_path):
    # runs ffprobe tool to get frame rate info in JSON
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "json",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    rate = info["streams"][0]["r_frame_rate"]
    # parses out frame rate and converts to float (25/1 -> 25.0 fps)
    num, denom = map(int, rate.split("/"))
    return num / denom

# If video is already 24fps, just copy to _24fps.mp4 (no conversion)
def copy_as_24fps(input_path, output_path):
    print(f"Copying {input_path} to {output_path} (already 24 FPS)")
    shutil.copy2(input_path, output_path)
    print("Done copying.")

# Converts video to 24fps and saves with _24fps.mp4
def convert_to_24fps(input_path, output_path):
    print(f"Converting {input_path} to 24 FPS as {output_path} ...")
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-r", "24",
        "-y",  # overwrite output if exists
        output_path
    ]
    # hides ffmpeg output for cleaner logs
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Done converting.")

# Optionally: extracts frames at 24fps using ffmpeg if not already 24fps
def extract_frames_24fps(input_path, output_folder):
    # make output folder if doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    # run ffmpeg to process video and save frames as png
    # hides most ffmpeg output for cleaner logs
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-vf", "fps=24",
        os.path.join(output_folder, "frame_%04d.png"),
        "-hide_banner", "-loglevel", "error"
    ]
    print(f"Extracting frames at 24 FPS from {input_path} to {output_folder} ...")
    subprocess.run(cmd, check=True)
    print("Done extracting frames.")

if __name__ == "__main__":
    video_folder = "Videos"
    for filename in os.listdir(video_folder):
        # look for .mp4
        if filename.endswith(".mp4") and "_24fps" not in filename:
            input_path = os.path.join(video_folder, filename)
            fps = get_fps(input_path)
            print(f"{filename}: Detected FPS = {fps:.2f}")
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(video_folder, f"{name}_24fps{ext}")
            if abs(fps - 24) < 0.05:
                # already in 24fps, just copy to _24fps file (FAST)
                copy_as_24fps(input_path, output_path)
            else:
                # not 24fps, so convert it (SLOW)
                convert_to_24fps(input_path, output_path)
