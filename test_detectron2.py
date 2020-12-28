""" Author: Oluwatosin Olubiyi
    
    Github Username: Olubiyiontheweb 
    
    COCO dataset and MaskRCNN (Facebook Detectron2) in a image"""

from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer, GenericMask
from detectron2 import model_zoo
from detectron2.structures import instances

import cv2
import numpy as np
import requests
import os

# Load an image
# res = requests.get("https://live.staticflickr.com/700/33224654191_fdaee2e3f1_c_d.jpg")
# image = np.asarray(bytearray(res.content), dtype="uint8")

# image = cv2.imdecode(image, cv2.IMREAD_COLOR)

# Root directory of the project
ROOT_DIR = os.path.abspath("../deep_learning_practices/pytouch_model_coco/")

image = cv2.imread(os.path.join(ROOT_DIR, "33224654191_fdaee2e3f1_c.jpg"))

cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75  # Threshold
cfg.MODEL.WEIGHTS = os.path.join(ROOT_DIR, "model_final_a54504.pkl")
cfg.MODEL.DEVICE = "cpu"  # cpu or cuda

print(cfg.MODEL.WEIGHTS)

# Create predictor
predictor = DefaultPredictor(cfg)

# Make prediction
output = predictor(image)


classes = output["instances"].pred_classes
nclasses = classes.numpy()
pos = np.where(nclasses == 0)[0][0]
# print(type(int(pos)))

# masks = np.asarray(output["instances"][int(pos)].pred_masks)
# print(image.shape)
# masks = [GenericMask(x, image.shape[0], image.shape[1]) for x in masks]

print("Mask below")
print(output["instances"][int(pos)].pred_masks.numpy().shape)
points = classes = output["instances"][int(pos)].pred_masks.numpy()
print(points.shape)
# points = np.reshape(points, (1, 696, 800))
points = np.squeeze(points, axis=0)
print(points.shape)
print(image.shape)
points = np.argwhere(points == True)
print(points)

mask = np.zeros((image.shape[0], image.shape[1]))

cv2.fillConvexPoly(mask, points, 1)

mask = mask.astype(np.bool)

out = np.zeros_like(image)
out[mask] = image[mask]

# masks = Visualizer._convert_masks(masks)

# print(masks)

# print(output["instances"])

# instance = Instances(image_size=(480, 640))
# instance.set("pred_classes", labels)
# print(instance)

v = Visualizer(
    image[:, :, ::-1],
    scale=0.8,
    metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
    instance_mode=ColorMode.IMAGE,
)
v = v.draw_instance_predictions(output["instances"].to("cpu"))

box = output["instances"][int(pos)].pred_boxes
startX, startY, endX, endY = box.tensor.numpy().astype("int").tolist()[0]
# print(startX)
# startX, startY, endX, endY = output["instances"][int(pos)].pred_boxes
detected_person = v.get_image()[:, :, ::-1][
    startY + 1 : endY + 1, startX - 10 : endX + 1
]

# print(dir(v))
cv2.imshow("images", v.get_image()[:, :, ::-1])
cv2.imwrite("segmented.jpg", v.get_image()[:, :, ::-1])


cv2.imshow("detected_person", detected_person)
cv2.imwrite("segmented_cropped.jpg", detected_person)

cv2.imshow("Extracted Image", out)
cv2.imwrite("segmented_bg_removed.jpg", out)

cv2.waitKey(0)
