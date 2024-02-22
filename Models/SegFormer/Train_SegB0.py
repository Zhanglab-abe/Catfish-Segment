# %% [markdown]
# # 指南
# https://huggingface.co/docs/transformers/tasks/semantic_segmentation

# %%
# %env CUDA_DEVICE_ORDER=PCI_BUS_ID
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
# from datasets import load_dataset
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import wandb
import evaluate
wandb.init(project="segb0", entity="msu-zhanglab")
from torchvision.transforms.functional import crop, resize



# %%
from datasets import load_dataset, Image, Dataset
import sys
import os
images = []
masks = []

for root, dirs, files in os.walk(f"{sys.path[0]}/data/images/training/"):
    # print(root, dirs, files)
    files.sort()
    for file in files:
        images.append(f"{root}{file}")
    break
for root, dirs, files in os.walk(f"{sys.path[0]}/data/annotations/training/"):
    # print(root, dirs, files)
    files.sort()
    for file in files:
        masks.append(f"{root}{file}")
    break

train_ds = Dataset.from_dict({"image": images,"annotation":masks}).cast_column("image", Image()).cast_column("annotation", Image())


# %%

images = []
masks = []

for root, dirs, files in os.walk(f"{sys.path[0]}/data/images/validation/"):
    # print(root, dirs, files)
    files.sort()
    for file in files:
        images.append(f"{root}{file}")
    break
for root, dirs, files in os.walk(f"{sys.path[0]}/data/annotations/validation/"):
    # print(root, dirs, files)
    files.sort()
    for file in files:
        masks.append(f"{root}{file}")
    break

test_ds = Dataset.from_dict({"image": images,"annotation":masks}).cast_column("image", Image()).cast_column("annotation", Image())


# %%
print(train_ds)
print(test_ds)

# %%
# print(ds)

tempdata = train_ds[2]

print(tempdata)
# tempdata["image"].show()
# tempdata["annotation"].show()

# %%
id2label={'0': 'background', '1': 'Head', '2': 'Body', '3': 'Fins', '4': 'Tail'}
id2label = {int(k): v for k, v in id2label.items()}
print(id2label)
label2id = {v: k for k, v in id2label.items()}
print(label2id)
num_labels = len(id2label)

# %%
from transformers import AutoFeatureExtractor
# from transformers import SegformerFeatureExtractor, SegformerForImageClassification


pretrained_model_name = "nvidia/mit-b0"

feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name, reduce_labels=True)

# %%
from transformers import AutoModelForSemanticSegmentation

# pretrained_model_name = "nvidia/mit-b0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use GPU if available, otherwise use a CPU

model = AutoModelForSemanticSegmentation.from_pretrained(
    pretrained_model_name, id2label=id2label, label2id=label2id
).to(device)

# %%
from torchvision.transforms import ColorJitter

jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)

# %%
def train_transforms(example_batch):
    images = [jitter(x) for x in example_batch["image"]]
    labels = [x for x in example_batch["annotation"]]
    cropped_images = []
    cropped_labels = []
    for img, lbl in zip(images, labels):
        # crop to 640x640
        top = (img.size[1] - 640) // 2
        left = (img.size[0] - 640) // 2
        img = crop(img, top, left, 640, 640)
        lbl = crop(lbl, top, left, 640, 640)
        cropped_images.append(img)
        cropped_labels.append(lbl)
    inputs = feature_extractor(cropped_images, cropped_labels, return_tensors="pt")
    return inputs


def val_transforms(example_batch):
    images = [jitter(x) for x in example_batch["image"]]
    labels = [x for x in example_batch["annotation"]]
    cropped_images = []
    cropped_labels = []
    for img, lbl in zip(images, labels):
        # crop to 640x640
        top = (img.size[1] - 640) // 2
        left = (img.size[0] - 640) // 2
        img = crop(img, top, left, 640, 640)
        lbl = crop(lbl, top, left, 640, 640)
        cropped_images.append(img)
        cropped_labels.append(lbl)
    inputs = feature_extractor(cropped_images, cropped_labels, return_tensors="pt")
    return inputs

# %%
train_ds.set_transform(train_transforms)
test_ds.set_transform(val_transforms)

# %%
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="segformer-b0-mydataSet",
    learning_rate=6e-5,
    num_train_epochs=50,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    save_total_limit=5,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=20,
    eval_steps=20,
    logging_steps=1,
    eval_accumulation_steps=5,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to="wandb"
    )

# %%
# import evaluate

metric = evaluate.load("mean_iou")
metric_f1 = evaluate.load("f1")
metric_aacc = evaluate.load("accuracy")
# # %%
def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        labelsFlatten = labels.flatten()
        logits_tensor = torch.from_numpy(logits)
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)
        pred_labels = logits_tensor.detach().cpu().numpy()
        pred_labelsFlatten = pred_labels.flatten()
        metrics = metric.compute(
            predictions=pred_labels,
            references=labels,
            num_labels=num_labels,
            ignore_index=255,
            # reduce_labels=True,
        )
        metrics_f1_macro = metric_f1.compute(
            predictions=pred_labelsFlatten,
            references=labelsFlatten,
            average="macro"
        )
        metrics_f1_micro = metric_f1.compute(
            predictions=pred_labelsFlatten,
            references=labelsFlatten,
            average="micro"
        )
        metrics_f1_none = metric_f1.compute(
            predictions=pred_labelsFlatten,
            references=labelsFlatten,
            average=None
        )
        metrics_aacc = metric_aacc.compute(
            predictions=pred_labelsFlatten,
            references=labelsFlatten,
            #ignore_index=255
        )
        for key, value in metrics.items():
            if type(value) is np.ndarray:
                metrics[key] = value.tolist()
        metrics["f1_macro"] = metrics_f1_macro["f1"]
        metrics["f1_micro"] = metrics_f1_micro["f1"]
        metrics["f1_none"] = metrics_f1_none["f1"].tolist()
        for i in range(len(metrics["f1_none"])):
            metrics[f"f1_none_{i}"] = metrics["f1_none"][i]
        metrics["aAcc"] = metrics_aacc["accuracy"]
        # print(metrics)
        return metrics

# %%
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)

# %%
trainer.train()

# %%
wandb.finish()
