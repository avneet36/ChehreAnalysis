import subprocess
import os

def convert_to_24fps(input_path, output_path=None):
    # If no output filename is given, create one automatically
    if output_path is None:
        name, ext = os.path.splitext(input_path)
        output_path = f"{name}_24fps{ext}"
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-r", "24",
        "-y",  # overwrite output if exists
        output_path
    ]
    print(f"Converting {input_path} to 24 FPS as {output_path} ...")
    subprocess.run(cmd, check=True)
    print("Done.")

if __name__ == "__main__":
    video_folder = "Videos"  # Your folder name
    for filename in os.listdir(video_folder):
        if filename.endswith(".mp4") and "_24fps" not in filename:
            input_path = os.path.join(video_folder, filename)
            convert_to_24fps(input_path)
