import dlib
import cv2

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("PATH TO shape_predictor_68_face_landmarks.dat")

def align_video(video_path, output_path):
    """
    Aligns all frames in a video based on the first detected face.

    Parameters:
        video_path (str): Path to the input video file.
        output_path (str): Path to save the aligned output video.

    Returns:
        None: The aligned video is saved to output_path.
    """
    cap = cv2.VideoCapture(video_path)
    print(f"Number of frames loaded: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, fourcc, fps, (512, 512))

    M = None

    ret, frame = cap.read()
        
    is_vertical = False
    if ret:
        cv2.imwrite("first_frame.jpg", frame)
        h, w = frame.shape[:2]
        if h < w:  # If height < width, it's likely a rotated vertical video
            is_vertical = True
            print("Detected vertically recorded video that's been rotated")
            # Rotate it back to its original orientation
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        print(f"Detected {len(faces)} face(s) in the first frame")
        if len(faces) > 0:
            # Apply function to detected face, like detecting keypoints instead of aligning the frame
            M = align_face(frame, faces[0])
        # Reset the video to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if M is not None:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if is_vertical:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            aligned = align_frame(frame, M)
            out.write(aligned)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
