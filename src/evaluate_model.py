import pandas as pd
import numpy as np
import argparse
import os
from transformers import AutoModelForImageClassification, Trainer
from datasets import load_from_disk, Dataset
from sklearn.metrics import accuracy_score, classification_report, det_curve
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import describe, gaussian_kde
from auxiliary_utils import Auxiliary
import cv2 
from evaluate import load

##### ARGS #####
parser = argparse.ArgumentParser(description='Evaluate the Vit')

parser.add_argument('--out-path', '-o', metavar="O", dest='out_path', type=str, help='Job out path')
parser.add_argument('--data-set', '-d', metavar="D", dest='dataset', type=str, help='Job out path')
parser.add_argument('--run-inference', "-r", action="store_true", help='Run inference if this flag is set')
args = parser.parse_args()

##### LOAD normal and processed dataset ####
dataset_normal = load_from_disk(f"{os.environ['HOME']}/biometrics/black-hole/datasets/{args.dataset}/")
dataset_processed = load_from_disk(f"{os.environ['HOME']}/biometrics/black-hole/processed_datasets/{args.dataset}/")
labels = dataset_processed['train'].features['label'].names
print(f"Labels of Dataset: {labels}")

#### Evaluate on test set #####

model = AutoModelForImageClassification.from_pretrained(f"{args.out_path}/model-saved/", output_attentions = True)

print(f"Args run was set to {args.run_inference}")

if args.run_inference:
    print("Starting inference")
    male_total_logits_list = []
    male_true_logits_list = []  #stores the score of the logits where the label male was true
    female_true_logits_list = [] # same for female
    labels_list = []

    model.eval()

    for entry in tqdm(dataset_processed["test"]):
        pixel_values = entry["pixel_values"].unsqueeze(0)  # Add batch dimension
        label = entry["labels"]

        with torch.no_grad():
            outputs = model(pixel_values)

        probs = F.softmax(outputs.logits, dim=1)[0]
        # select the Male probability
        male_total_logits_list.append(probs[1].item())
        # Append 0 if label is not, else append the probability
        if label == 1:
            male_true_logits_list.append(probs[1].item())  
        else:
            #print(probs)
            female_true_logits_list.append(1 - probs[0].item())  # 0 - Woman, 1 - Man
        # Append the label to the labels_list
        labels_list.append(label)

    np_male_list = np.array(male_true_logits_list)
    np_female_list = np.array(female_true_logits_list)
    np_total_list = np.array(male_total_logits_list)

    os.mkdir(f"{args.out_path}/scores/")
    stats = np.array(np_male_list)
    np.save(f"{args.out_path}/scores/male_scores.npy", stats)
    stats = np.array(np_female_list)
    np.save(f"{args.out_path}/scores/female_scores.npy", stats)
    stats = np.array(np_total_list)
    np.save(f"{args.out_path}/scores/total_scores.npy", stats)
    stats = np.array(labels_list)
    np.save(f"{args.out_path}/scores/labels_list.npy", stats)
else:
    print("Loading scores")
    np_total_list = np.load(f"{args.out_path}/scores/total_scores.npy")
    np_male_list = np.load(f"{args.out_path}/scores/male_scores.npy")  #stores the score of the logits where the label male was true
    np_female_list = np.load(f"{args.out_path}/scores/female_scores.npy") # same for female
    labels_list = np.load(f"{args.out_path}/scores/labels_list.npy")

os.mkdir(f"{args.out_path}/imgs/")

#### Save Attention maps to image
image_used = 100

img = dataset_normal["test"][image_used]["image"]
pixel_values = dataset_processed["test"][image_used]["pixel_values"].unsqueeze(0)  # Add batch dimension
model.eval()
with torch.no_grad():
    outputs = model(pixel_values)

att = torch.stack(outputs.attentions).squeeze(1)
# Average the attention weights across all heads.
att_mat = torch.mean(att, dim=1)

# To account for residual connections, we add an identity matrix to the
# attention matrix and re-normalize the weights.
residual_att = torch.eye(att_mat.size(1))
aug_att_mat = att_mat + residual_att
aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

# Recursively multiply the weight matrices
joint_attentions = torch.zeros(aug_att_mat.size())
joint_attentions[0] = aug_att_mat[0]

for n in range(1, aug_att_mat.size(0)):
    joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

# Attention from the output token to the input space.
v = joint_attentions[-1]
grid_size = int(np.sqrt(aug_att_mat.size(-1)))
mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
mask = cv2.resize(mask / mask.max(), img.size)[..., np.newaxis]
result = (mask * img).astype("uint8")

heatmap = cm.turbo(mask.squeeze(), alpha=0.9)  # jet colormap with alpha transparency
### IMPORTANT ----------------------
# With the dataset both eyes mirrored heatmap must have only 3 values in the last dimension
# where as for the other datasets it needs 4
if args.dataset == "both_eyes_together":
    dim = 3
else:
    dim = 4
                        # |
            ##Edit this   v
heatmap = (heatmap[:, :, :dim] * 255).astype(np.uint8)
alpha = 0.6
blended = (np.array(img) * (1 - alpha) + heatmap * alpha).astype(np.uint8)

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
ax1.set_title('Original')
ax2.set_title('Attention Map, Red High importance')
_ = ax1.imshow(img)
_ = ax2.imshow(blended)
plt.tight_layout()
plt.savefig(f"{args.out_path}/attention1.png")

#### #All 12 attention layers visualized
num_maps = len(joint_attentions)
fig, axes = plt.subplots(nrows=num_maps, ncols=2, figsize=(12, 6 * num_maps))

for i, v in enumerate(joint_attentions):
    # Attention from the output token to the input space.
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    mask = cv2.resize(mask / mask.max(), img.size)[..., np.newaxis]
    result = (mask * img).astype("uint8")

    heatmap = cm.turbo(mask.squeeze(), alpha=0.9)  # jet colormap with alpha transparency
    if args.dataset == "both_eyes_together":
        dim = 3
    else:
        dim = 4

    heatmap = (heatmap[:, :, :dim] * 255).astype(np.uint8)
    alpha = 0.6

    blended = (np.array(img) * (1 - alpha) + heatmap * alpha).astype(np.uint8)

    ax1, ax2 = axes[i]
    ax1.set_title('Original')
    ax2.set_title(f'Attention Map {i+1}')
    ax1.imshow(img)
    ax2.imshow(blended)
    ax1.axis("off")
    ax2.axis("off")
plt.tight_layout()
plt.savefig(f"{args.out_path}/attention12.png")

####################################
auxiliary = Auxiliary(np_male_list ,np_female_list, labels_list, np_total_list, 0.5, "percent", out_path=f"{args.out_path}/imgs")
auxiliary.kde_with_threshold()
auxiliary.plot_det()
auxiliary.print_classification_metrics()