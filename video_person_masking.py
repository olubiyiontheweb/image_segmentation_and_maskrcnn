""" Author: Oluwatosin Olubiyi
    
    Github Username: Olubiyiontheweb 
    
    COCO dataset and MaskRCNN in a video"""

import cv2
import sys
import os

# import skimage.io

# Root directory of the project
ROOT_DIR = os.path.abspath("../deep_learning_practices/Mask_RCNN")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import visualize
import mrcnn.model as modellib
from mrcnn import utils, config

# Import COCO config
# To find local version
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# COCO Class names
class_names = [
    "BG",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


class SimpleConfig(config.Config):
    NAME = "coco_inference"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    NUM_CLASSES = len(class_names)


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


# config = InferenceConfig()
# config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(
    mode="inference", model_dir=os.getcwd(), config=SimpleConfig()
)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

print("Everything is fine")

video_cam = cv2.VideoCapture(0)

if video_cam.isOpened() == False:
    print("unable to open camera")

while True:
    check, frame = video_cam.read()

    if check == False:
        print("Unable to read from camera")
        break

    # image = skimage.io.imread(frame)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run detection
    results = model.detect([frame], verbose=0)

    # Visualize results
    r = results[0]

    image = visualize.display_instances(
        image, r["rois"], r["masks"], r["class_ids"], class_names, r["scores"]
    )

    cv2.imshow("Video Camera", image)

    key = cv2.waitKey(1)

    if key == ord("q"):
        print("Quiting .....")
        break
    elif key == ord("w"):        
        print("Writing image to file .....")
        cv2.imwrite("segmented_image1.jpg", image)


cv2.destroyAllWindows()
