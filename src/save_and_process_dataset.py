from datasets import load_dataset
from transformers import ViTImageProcessor
import os
name = "both_eyes_together"
images = f'{os.environ["HOME"]}/biometrics/black-hole/{name}'
dataset = load_dataset('imagefolder', data_dir=images)

####################################################################################
# Takes an image folder already in the format of HuggingFace, loads it, stores the 
# loaded dataset, processes it and stores the processed dataset

 
### save normal dataset
ds = f"{os.environ['HOME']}/biometrics/black-hole/datasets/{name}"

print(f"Saving dataset to {ds}")
dataset.save_to_disk(ds)

if os.path.exists(ds):
    print(f"Processed dataset saved successfully to {os.path.abspath(ds)}")
else:
    print(f"Failed to save the processed dataset to {ds}. Please check for errors.")

### preprocesss
model_name_or_path = 'google/vit-base-patch16-224-in21k'
processor = ViTImageProcessor.from_pretrained(model_name_or_path)

def process_image(example):
    #Convert image to 3d from grayscale
    img = example["image"].convert("RGB")
    inputs = processor(img, return_tensors='pt')
    inputs["pixel_values"] = inputs["pixel_values"].squeeze(0)  # Remove the batch dimension
    inputs['labels'] = example["label"]
    #inputs_dict = {k: torch.from_numpy(v) for k, v in inputs.items()}
    return inputs

dataset_processed = dataset.map(process_image)
dataset_processed.set_format(type="torch", columns=["pixel_values", "labels"]) # .map removes tensors so put them back

print(dataset_processed)

#### save preprocessed datset
pds = os.path.join(f"{os.environ['HOME']}/biometrics/black-hole/processed_datasets", name)

print(f"Saving processed dataset to {pds}")
dataset_processed.save_to_disk(pds)

if os.path.exists(pds):
    print(f"Processed dataset saved successfully to {os.path.abspath(pds)}")
else:
    print(f"Failed to save the processed dataset to {pds}. Please check for errors.")
print("Done!")