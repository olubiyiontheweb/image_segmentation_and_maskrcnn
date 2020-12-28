""" Author: Oluwatosin Olubiyi
    
    Github Username: Olubiyiontheweb 
    
    COCO dataset and MaskRCNN (Facebook Detectron2) in a video"""

from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2 import model_zoo
import numpy as np
import cv2
import os


# Root directory of the project
ROOT_DIR = os.path.abspath("../deep_learning_practices/pytouch_model_coco/")

cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75  # Threshold
cfg.MODEL.WEIGHTS = os.path.join(ROOT_DIR, "model_final_a54504.pkl")
cfg.MODEL.DEVICE = "cpu"  # cpu or cuda
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model

# Create predictor
predictor = DefaultPredictor(cfg)

# COCO Class names
""" class_names = [
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
    "BG",
] """


print("Everything is fine")

video_cam = cv2.VideoCapture(0)

if video_cam.isOpened() == False:
    print("unable to open camera")

# Extract video properties
width = int(video_cam.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames_per_second = video_cam.get(cv2.CAP_PROP_FPS)
num_frames = int(video_cam.get(cv2.CAP_PROP_FRAME_COUNT))
readFrames = 0


while True:
    check, frame = video_cam.read()

    if check == False:
        print("Unable to read from camera")
        break

    # Make sure the frame is colored
    # gray_frame_copy = frame.copy()
    # gray_frame_copy = cv2.cvtColor(gray_frame_copy, cv2.COLOR_BGR2RGB)

    # Make prediction
    output = predictor(frame)

    try:
        # get first person detected

        print(output["instances"].pred_boxes)
        classes = output["instances"].pred_classes
        classes = classes.numpy()
        pos = np.where(classes == 0)[0][0]

        v = VideoVisualizer(
            metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            instance_mode=ColorMode.IMAGE,
        )

        v = v.draw_instance_predictions(frame, output["instances"][int(pos)].to("cpu"))

        box = output["instances"][int(pos)].pred_boxes
        startX, startY, endX, endY = box.tensor.numpy().astype("int").tolist()[0]
        detected_person = frame[startY:endY, startX:endX]

        cv2.imshow("images", v.get_image()[:, :, ::-1])

        # cv2.imshow("images", v.get_image()[:, :, ::-1])
    except:
        cv2.imshow("images", v.get_image()[:, :, ::-1])

    # if output["instances"]:
    #     # get first person detected
    #     classes = output["instances"].pred_classes
    #     classes = classes.numpy()
    #     pos = np.where(classes == 0)[0][0]

    #     v = VideoVisualizer(
    #         metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
    #         instance_mode=ColorMode.IMAGE,
    #     )

    #     v = v.draw_instance_predictions(frame, output["instances"][int(pos)].to("cpu"))

    #     cv2.imshow("images", v.get_image()[:, :, ::-1])
    # else:
    #     cv2.imshow("images", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        print("Quiting .....")
        break
    elif key == ord("w"):        
        print("Writing image to file .....")
        cv2.imwrite("segmented_framed1.jpg", v.get_image()[:, :, ::-1])

video_cam.release()
cv2.destroyAllWindows()
