from face_alignment import FaceAlignment, LandmarksType
from skimage import io
import matplotlib.pyplot as plt

# Initialize the face alignment detector
fa = FaceAlignment(LandmarksType.TWO_D, flip_input=False, device='cpu')

# Load your test image
image = io.imread("testImage.jpg")  # Make sure the file name matches exactly

# Run face alignment
landmarks = fa.get_landmarks(image)

# Visualize the results
if landmarks is not None:
    for face in landmarks:
        plt.imshow(image)
        plt.scatter(face[:, 0], face[:, 1], s=10)
        plt.title("Detected facial landmarks")
        plt.axis("off")
        plt.show()
else:
    print("No face detected.")
