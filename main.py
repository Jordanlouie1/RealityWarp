import cv2
import openai
import requests
from PIL import Image
import time
import os


# Replace with your OpenAI API key
openai.api_key = 'YOUR_API_KEY'

def capture_image(file_path):
    # Open a connection to the camera (0 is default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    # Capture frame
    ret, frame = cap.read()

    if ret:
        # Save the captured image to the specified file path
        cv2.imwrite(file_path, frame)
        print(f"Image captured and saved to {file_path}")
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

def show_and_close_image(image_path, display_time=5):
    # Load the image
    image = Image.open(image_path)

    # Display the image
    image.show()

    # Wait for the specified time (in seconds)
    time.sleep(display_time)

    # Forcibly close the image by terminating the viewer (works for most OS image viewers)
    if os.name == 'nt':  # For Windows
        os.system('taskkill /IM Microsoft.Photos.exe /F')
    elif os.name == 'posix':  # For macOS and Linux
        os.system('killall Preview')  # For macOS (Preview app)
        os.system('pkill display')  # For Linux (ImageMagick 'display' command)



if __name__ == "__main__":
    t = 0
    while(t < 1):
        # Path to save the image
        image_file_path = "captured_image.jpg"

        #Capture the image
        capture_image(image_file_path)

        #Describe image as an animal    animal

        #animal = describe_image(image_file_path)

        #Display image with applied filter
        #captured image = animal
        #figure out how to close image - jordan
        show_and_close_image(image_file_path, display_time=5)  # Display image for 5 seconds

    #Livekit

