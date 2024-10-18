from flask import Flask, Response
import cv2
import threading
import requests
from PIL import Image
import time
import os
import torch
import clip
from PIL import Image
import cv2
import numpy as np
from flask_cors import CORS
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

current_animal = "Dog"
lock = threading.Lock()

# Load the CLIP model and the preprocessing function
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


# Animal types
PREVIEW = 0       # Preview mode
DOG = 8
SNAKE = 9
BIRD = 10
HUMAN = 11
INSECT = 12
ELEPHANT = 13


camera1 = 0  # for default camera
camera2 = 1  # for external camera

camera = cv2.VideoCapture(1)

# Set default filter mode to preview
image_filter = SNAKE

mode_names = {
    PREVIEW: "Preview",
    DOG: "Dog",
    SNAKE: "Snake",
    BIRD: "Bird",
    HUMAN: "Human",
    INSECT: "Insect",
    ELEPHANT: "Elephant"
}

def capture_image(file_path):
    # Open a connection to the camera (0 is default camera)
    cap = cv2.VideoCapture(camera1)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    # Capture frame
    ret, frame = cap.read()

    if ret:
        # Save the captured image to the specified file path
        cv2.imwrite(file_path, frame)
        # print(f"Image captured and saved to {file_path}")
    else:
        print("Failed to capture image")

    # Release the camera
    cap.release()

def describe_image(file_path):
        if file_path != None:
            print("Image Description:")
            animal = "filepath output"
        else:
            print("Could not describe the image")
        return animal

def classify_pose(image_path):
    # Load and preprocess the image
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # Define the possible pose labels and their corresponding animals
    pose_options = [
        ("a person posing like a bendy stretchy snake", "Snake"),
        ("a person posing with both arms stretched out like wings of a bird", "Bird"),
        ("a person posing like a dog on all fours", "Dog"),
        ("a person with one arm stretched out in front of their face", "Elephant")
    ]

    # Encode text prompts with the CLIP model
    text_inputs = torch.cat([clip.tokenize(f"This is {label[0]}") for label in pose_options]).to(device)

    # Compute image and text features using CLIP
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_inputs)

    # Normalize features for cosine similarity comparison
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute cosine similarity between the image and text features
    similarities = (image_features @ text_features.T).squeeze(0)

    # Find the text with the highest similarity to the image
    best_match_idx = similarities.argmax().item()

    # Return the animal corresponding to the best matching text
    return pose_options[best_match_idx][1]

def show_and_close_image(image_path, display_time=5):
    # Load the image
    image = Image.open(image_path)

    # Display the image
    image.show()

    # Wait for the specified time (in seconds)
    time.sleep(display_time)
    os.system('pkill display')
    # Forcibly close the image by terminating the viewer (works for most OS image viewers)
    if os.name == 'nt':  # For Windows
        os.system('taskkill /IM Microsoft.Photos.exe /F')
    elif os.name == 'posix':  # For macOS and Linux
        os.system('killall Preview')  # For macOS (Preview app)
        os.system('pkill display')  # For Linux (ImageMagick 'display' command)

def list_cameras():
    """
    Lists all available cameras and returns their indices.
    """

    index = 0
    cameras = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            break
        cameras.append(index)
        cap.release()
        index += 1

    return cameras

available_cameras = list_cameras()
print(f"Available cameras: {available_cameras}")

def generate_output_frames():
    while True:
        global current_animal 
        # Capture frame-by-frame
        success, frame = camera.read()
        result = None
        if not success:
            break
        else:


            if current_animal == "Snake":
                blurred = cv2.GaussianBlur(frame, (35, 35), 0)

                # 2. Convert to grayscale (snakes are not sensitive to color)
                gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

                # 3. Apply a heatmap-like effect to simulate heat vision
                # Use color mapping (COLORMAP_JET mimics infrared heat detection)
                heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

                # 4. Apply Canny edge detection to simulate motion detection or sensing contours
                edges = cv2.Canny(gray, 50, 150)
                edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Convert edges to 3 channels

                # 5. Combine the heatmap and edges for a final snake-vision-like effect
                result = cv2.addWeighted(heatmap, 0.8, edges_colored, 0.2, 0)

            elif current_animal == "Dog":
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

                # 2. Split LAB channels (L: lightness, A: green to red, B: blue to yellow)
                l, a, b = cv2.split(lab)

                # 3. Shift the color channels to simulate dichromatic vision (blue/yellow only)
                # Reduce the A channel (green-red) to simulate the lack of red perception
                a[:] = 128  # Neutralize red-green sensitivity
                # Increase the B channel (blue-yellow) slightly to enhance the yellow-blue vision
                b[:] = np.clip(b * 1.2, 0, 255)

                # 4. Merge the modified channels back
                modified_lab = cv2.merge([l, a, b])

                # 5. Convert back to BGR color space
                dog_color_vision = cv2.cvtColor(modified_lab, cv2.COLOR_LAB2BGR)

                # 6. Apply a slight Gaussian blur to simulate lower acuity
                blurred = cv2.GaussianBlur(dog_color_vision, (11, 11), 0)

                # 7. Adjust contrast and brightness (dogs see better in low light)
                contrast = 1.2  # Increase contrast
                brightness = 10  # Increase brightness slightly

                adjusted_frame = cv2.convertScaleAbs(blurred, alpha=contrast, beta=brightness)

                result = adjusted_frame

            elif current_animal == "Insect":
                # Step 1: Apply fisheye effect to simulate wide-angle insect vision

                pixel_size = 20
                contrast = 1.5
                brightness = 0

                height, width = frame.shape[:2]
                K = np.array([[width, 0, width // 2],
                              [0, height, height // 2],
                              [0, 0, 1]])
                D = np.array([-0.4, 0.2, 0, 0])  # Distortion coefficients
                map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K, (width, height), 5)
                fisheye_image = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

                # Step 2: Pixelate the image to simulate the compound eye's pixelated vision
                small = cv2.resize(fisheye_image, (width // pixel_size, height // pixel_size),
                                   interpolation=cv2.INTER_LINEAR)
                pixelated = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)

                # Step 3: Simulate UV light vision by shifting colors to blue/violet tones
                hsv = cv2.cvtColor(pixelated, cv2.COLOR_BGR2HSV)
                hsv[:, :, 0] = (hsv[:, :, 0] + 40) % 180  # Shift hue channel to simulate UV vision
                uv_vision = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                # Step 4: Adjust contrast and brightness to simulate motion sensitivity
                final_frame = cv2.convertScaleAbs(uv_vision, alpha=contrast, beta=brightness)

                result = final_frame

            elif current_animal == "Elephant":
                # Step 1: Convert to the LAB color space (to work with lightness and color channels)
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

                # Step 2: Modify color channels to simulate dichromatic (blue/yellow) vision
                l, a, b = cv2.split(lab)

                # Neutralize the red-green sensitivity (set the A channel to a neutral value)
                a[:] = 128

                # Enhance the blue-yellow perception (increase B channel)
                b[:] = np.clip(b * 1.1, 0, 255)

                # Merge the modified LAB channels back
                lab_modified = cv2.merge([l, a, b])

                # Convert back to BGR color space for further processing
                color_adjusted = cv2.cvtColor(lab_modified, cv2.COLOR_LAB2BGR)

                # Step 3: Apply Gaussian blur to simulate blurry vision
                blurred = cv2.GaussianBlur(color_adjusted, (15, 15), 0)

                # Step 4: Lower brightness to simulate reduced daytime vision sensitivity
                brightness = -30
                result = cv2.convertScaleAbs(blurred, beta=brightness)

            elif current_animal == "Bird":
                # Step 1: Simulate tetrachromatic (enhanced color) vision by boosting saturation and brightness
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # Increase saturation and brightness
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)  # Saturation (color intensity)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.3, 0, 255)  # Brightness

                # Step 2: Simulate UV light perception by shifting hue
                hsv[:, :, 0] = (hsv[:, :, 0] + 20) % 180  # Shift hue towards violet/blue

                # Convert back to BGR after color adjustments
                color_adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                # Step 3: Sharpen the image to reflect high visual acuity
                kernel = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])  # Sharpening kernel
                sharpened = cv2.filter2D(color_adjusted, -1, kernel)

                # Step 4: Apply a slight fisheye effect to simulate a wide-angle field of view
                height, width = sharpened.shape[:2]
                K = np.array([[width, 0, width // 2],
                              [0, height, height // 2],
                              [0, 0, 1]])
                D = np.array([-0.2, 0.1, 0, 0])  # Distortion coefficients for a subtle fisheye effect
                map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K, (width, height), 5)
                result = cv2.remap(sharpened, map1, map2, interpolation=cv2.INTER_LINEAR)
            
            else:
                result = frame


            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', result)
            result = buffer.tobytes()

            # Yield the frame in a format suitable for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + result + b'\r\n')


          
def capture_video():
    global current_animal
    while True:
         # Path to save the image
        image_file_path = "captured_image.jpg"

        #Capture the image
        capture_image(image_file_path)

        #Describe image as an animal    animal

        pose = classify_pose(image_file_path)

        with lock:
            current_animal = pose

        print(f"YOU ARE A BEAUTIFUL {pose}")
        #show_and_close_image(image_file_path, display_time=5)  # Display image for 5 seconds

        time.sleep(1)



@app.route('/video_feed')
def video_feed():
    return Response(generate_output_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/current_animal')
def get_current_animal():
    with lock:
        return {"animal": current_animal}

if __name__ == "__main__":
    # Starte den Videoaufnahme-Thread
    video_thread = threading.Thread(target=capture_video)
    video_thread.start()


    app.run(debug=True)  # This will run on http://127.0.0.1:5000 by default