from datasets import load_dataset, Image, Dataset
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from PIL import ImageOps

my_pretrained_model_name = "/home/agricoptics/Desktop/CatFish/segformer/segformer-b0-mydataSet/checkpoint-7200"
pretrained_model_name = "nvidia/mit-b0"

images = []
masks = []

for root, dirs, files in os.walk(f"{sys.path[0]}/data/test/data_raw/images_raw/"):
    # print(root, dirs, files)
    files.sort()
    for file in files:
        images.append(f"{root}{file}")
    break

test_ds = Dataset.from_dict({"image": images}).cast_column("image", Image())

from transformers import AutoFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name, reduce_labels=True)

from PIL import Image
# image = test_ds[18]["image"]
# image = test_ds[64]["image"]
image = test_ds[14]["image"]
print(image)


width = 640
height = 640
# (width, height) = (image.width // 5, image.height // 5)
im_resized = image.resize((width, height))
im_resized.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use GPU if available, otherwise use a CPU
encoding = feature_extractor(image, return_tensors="pt")
pixel_values = encoding.pixel_values.to(device)

id2label={'0': 'background', '1': 'Head', '2': 'Body', '3': 'Fins', '4': 'Tail'}
id2label = {int(k): v for k, v in id2label.items()}
print(id2label)
label2id = {v: k for k, v in id2label.items()}
print(label2id)
num_labels = len(id2label)

from transformers import AutoModelForSemanticSegmentation

model = AutoModelForSemanticSegmentation.from_pretrained(
    my_pretrained_model_name, id2label=id2label, label2id=label2id
).to(device)


outputs = model(pixel_values=pixel_values)
logits = outputs.logits.cpu()

upsampled_logits = nn.functional.interpolate(
    logits,
    size=image.size[::-1],
    mode="bilinear",
    align_corners=False,
)

pred_seg = upsampled_logits.argmax(dim=1)[0]

# ade_palette = [[0,255,255],[255,255,0],[255,0,0],[0,255,0],[0,0,255]]
ade_palette = [[0,255,255],[255,255,0],[0,0,255],[0,255,0],[255,0,0]]

import matplotlib.pyplot as plt

color_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
palette = np.array(ade_palette)
for label, color in enumerate(palette):
    color_seg[pred_seg == label, :] = color
color_seg = color_seg[..., ::-1]  # convert to BGR

# Crop the segmentation map to the original image size
color_seg = Image.fromarray(color_seg)
color_seg = ImageOps.crop(color_seg, border=image.size[1] - color_seg.size[1])

# Combine the original image and the cropped segmentation map
img = np.array(image) * 0.5 + np.array(color_seg) * 0.5
img = img.astype(np.uint8)
img = Image.fromarray(img)
img.show()  # display the cropped image







