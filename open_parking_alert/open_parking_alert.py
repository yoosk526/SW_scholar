import os
import numpy as np
import cv2
import mrcnn.config
import mrcnn.utils
from mrcnn.model import MaskRCNN
from pathlib import Path

# For server++++++++++++++++++++
import requests
import json

# For timestamp
import time

url = "https://dev.rs-aiot.com/api/v1/integrations/http/f65b368b-370c-7f17-a97b-aa18d6b6ca44"
# url = "http://localhost:3000/api/user"
headers = {'Content-type': 'application/json; charset=utf-8'}

# Configuration that will be used by the Mask-RCNN library


class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class
    DETECTION_MIN_CONFIDENCE = 0.6

# Filter a list of Mask R-CNN detection results to get only the detected cars / trucks


def get_car_boxes(boxes, class_ids):
    car_boxes = []

    for i, box in enumerate(boxes):
        # If the detected object isn't a car / truck, skip it
        if class_ids[i] in [3, 8, 6]:
            car_boxes.append(box)

    return np.array(car_boxes)

# Variable for timestamp
now = time


# Root directory of the project
ROOT_DIR = Path(".")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# Video file or camera to process - set this to 0 to use your webcam instead of a video file
VIDEO_SOURCE = "parking_lot_5.mp4"

# Create a Mask-RCNN model in inference mode
model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())

# Load pre-trained model
model.load_weights(COCO_MODEL_PATH, by_name=True)

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# Location of parking spaces
parked_car_boxes = None

# Load the video file we want to run detection on
video_capture = cv2.VideoCapture(VIDEO_SOURCE)

# How many frames of video we've seen in a row with a parking space open
free_space_frames = 0

# Loop over each frame of video
while video_capture.isOpened():
    success, frame = video_capture.read()
    if not success:
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color
    rgb_image = frame[:, :, ::-1]

    # Run the image through the Mask R-CNN model to get results.
    results = model.detect([rgb_image], verbose=0)

    # Mask R-CNN assumes we are running detection on multiple images.
    # We only passed in one image to detect, so only grab the first result.
    r = results[0]

    # The r variable will now have the results of detection:
    # - r['rois'] are the bounding box of each detected object
    # - r['class_ids'] are the class id (type) of each detected object
    # - r['scores'] are the confidence scores for each detection
    # - r['masks'] are the object masks for each detected object (which gives you the object outline)

    if parked_car_boxes is None or len(parked_car_boxes) == 0:
        # This is the first frame of video - assume all the cars detected are in parking spaces.
        # Save the location of each car as a parking space box and go to the next frame of video.

        parked_car_boxes = get_car_boxes(r['rois'], r['class_ids'])
        for box in parked_car_boxes:
            print("Car: ", box)
        print(" parked_ Array Dimension = ", len(parked_car_boxes))

        # make a comparison array for checking free space
        comparison = np.zeros((len(parked_car_boxes)), dtype='bool')

        # array for storing timestamp of each parking lot area
        park_in = np.full((len(parked_car_boxes)), int(now.time()), dtype='int')
        park_out = np.zeros((len(parked_car_boxes)), dtype='int')

        data = {
            "deviceName": "kk-B-Park\n",
            "deviceType": "IDLEPARK\n",
            "timestamp": int(now.time()),
            "payload": {
                "A1": {"isfree": "False", "inTime": int(park_in[0]), "outTime": int(park_out[0])},
                "A2": {"isfree": "False", "inTime": int(park_in[1]), "outTime": int(park_out[1])},
                "A3": {"isfree": "False", "inTime": int(park_in[2]), "outTime": int(park_out[2])},
                "A4": {"isfree": "False", "inTime": int(park_in[3]), "outTime": int(park_out[3])},
                "A5": {"isfree": "False", "inTime": int(park_in[4]), "outTime": int(park_out[4])},
                "A6": {"isfree": "False", "inTime": int(park_in[5]), "outTime": int(park_out[5])},
                "A7": {"isfree": "False", "inTime": int(park_in[6]), "outTime": int(park_out[6])},
                "A8": {"isfree": "False", "inTime": int(park_in[7]), "outTime": int(park_out[7])},
                "A9": {"isfree": "False", "inTime": int(park_in[8]), "outTime": int(park_out[8])},
                "A10": {"isfree": "False", "inTime": int(park_in[9]), "outTime": int(park_out[9])},
                "A11": {"isfree": "False", "inTime": int(park_in[10]), "outTime": int(park_out[10])},
                "A12": {"isfree": "False", "inTime": int(park_in[11]), "outTime": int(park_out[11])}
            }
        }
        res = requests.post(url, headers=headers, data=json.dumps(data))
    else:
        # We already know where the parking spaces are. Check if any are currently unoccupied.

        # Get where cars are currently located in the frame
        car_boxes = get_car_boxes(r['rois'], r['class_ids'])

        for box in car_boxes:
            print("Car: ", box)
        print("Array Dimension = ", len(car_boxes))

        # See how much those cars overlap with the known parking spaces
        overlaps = mrcnn.utils.compute_overlaps(parked_car_boxes, car_boxes)

        # Assume no spaces are free until we find one that is free
        free_space = np.zeros((len(parked_car_boxes)), dtype='bool')
        # free_space = False

        i = 0
        # Loop through each known parking space box
        for parking_area, overlap_areas in zip(parked_car_boxes, overlaps):

            # For this parking space, find the max amount it was covered by any
            # car that was detected in our image (doesn't really matter which car)
            max_IoU_overlap = np.max(overlap_areas)

            # Get the top-left and bottom-right coordinates of the parking area
            y1, x1, y2, x2 = parking_area

            # Check if the parking space is occupied by seeing if any car overlaps
            # it by more than 0.20 using IoU
            if max_IoU_overlap < 0.40:
                # Parking space not occupied! Draw a green box around it
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                # Flag that we have seen at least one open space
                free_space[i] = True
                park_out[i] = int(now.time())
                i += 1
                # free_space = True
            else:
                # Parking space is still occupied - draw a red box around it
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

            # Write the IoU measurement inside the box
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, f"{max_IoU_overlap:0.2}", (x1 + 6, y2 - 6), font, 0.3, (255, 255, 255))
        
        print("주차 가능 공간 : ", free_space)

        data = {
            "deviceName": "kk-B-Park\n",
            "deviceType": "IDLEPARK\n",
            "timestamp": int(now.time()),
            "payload": {
                # "A1": {"isfree": np.array2string(free_space[3]), "inTime": 1662094009000, "outTime": 0},
                # "A2": {"isfree": np.array2string(free_space[0]), "inTime": 1662094009000, "outTime": 0},
                # "A3": {"isfree": np.array2string(free_space[4]), "inTime": 1662094009000, "outTime": 0},
                # "A4": {"isfree": np.array2string(free_space[5]), "inTime": 1662094009000, "outTime": 0},
                # "A5": {"isfree": np.array2string(free_space[2]), "inTime": 1662094009000, "outTime": 0},
                # "A6": {"isfree": np.array2string(free_space[1]), "inTime": 1662094009000, "outTime": 0},
                # "A7": {"isfree": np.array2string(free_space[6]), "inTime": 1662094009000, "outTime": 0},
                # "A8": {"isfree": np.array2string(free_space[7]), "inTime": 1662094009000, "outTime": 0},
                # "A9": {"isfree": np.array2string(free_space[8]), "inTime": 1662094009000, "outTime": 0},
                # "A10": {"isfree": np.array2string(free_space[9]), "inTime": 1662094009000, "outTime": 0},
                # "A11": {"isfree": np.array2string(free_space[10]), "inTime": 1662094009000, "outTime": 0},
                # "A12": {"isfree": np.array2string(free_space[11]), "inTime": 1662094009000, "outTime": 0}
                "A1": {"isfree": np.array2string(free_space[0]), "inTime": int(park_in[0]), "outTime": int(park_out[0])},
                "A2": {"isfree": np.array2string(free_space[1]), "inTime": int(park_in[1]), "outTime": int(park_out[1])},
                "A3": {"isfree": np.array2string(free_space[2]), "inTime": int(park_in[2]), "outTime": int(park_out[2])},
                "A4": {"isfree": np.array2string(free_space[3]), "inTime": int(park_in[3]), "outTime": int(park_out[3])},
                "A5": {"isfree": np.array2string(free_space[4]), "inTime": int(park_in[4]), "outTime": int(park_out[4])},
                "A6": {"isfree": np.array2string(free_space[5]), "inTime": int(park_in[5]), "outTime": int(park_out[5])},
                "A7": {"isfree": np.array2string(free_space[6]), "inTime": int(park_in[6]), "outTime": int(park_out[6])},
                "A8": {"isfree": np.array2string(free_space[7]), "inTime": int(park_in[7]), "outTime": int(park_out[7])},
                "A9": {"isfree": np.array2string(free_space[8]), "inTime": int(park_in[8]), "outTime": int(park_out[8])},
                "A10": {"isfree": np.array2string(free_space[9]), "inTime": int(park_in[9]), "outTime": int(park_out[9])},
                "A11": {"isfree": np.array2string(free_space[10]), "inTime": int(park_in[10]), "outTime": int(park_out[10])},
                "A12": {"isfree": np.array2string(free_space[11]), "inTime": int(park_in[11]), "outTime": int(park_out[11])}
            }
        }
        res = requests.post(url, headers=headers, data=json.dumps(data))

        # If at least one space was free, start counting frames
        # This is so we don't alert based on one frame of a spot being open.
        # This helps prevent the script triggered on one bad detection.
        if np.all(free_space == comparison):
            free_space_frames = 0
        else:
            free_space_frames += 1

        '''
        if free_space:
            free_space_frames += 1
        else:
            # If no spots are free, reset the count
            free_space_frames = 0
        '''

        # # If a space has been free for several frames, we are pretty sure it is really free!
        if free_space_frames > 4:
            # Write SPACE AVAILABLE!! at the top of the screen
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, f"SPACE AVAILABLE!", (10, 150), font, 3.0, (0, 255, 0), 2, cv2.FILLED)

        # Show the frame of video on the screen
        cv2.imshow('Video', frame)

    # Hit 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up everything when finished
video_capture.release()
cv2.destroyAllWindows()
