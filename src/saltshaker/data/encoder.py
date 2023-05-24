#!/usr/bin/env python3
import os
import pickle

import torch
import torchvision
from diffusers import AutoencoderKL
from PIL import Image

import saltshaker.shared as shared
from saltshaker.data.filedisk_loader import AspectBucket


def encode(model, buckets, config, progress):
    device = torch.device("cuda")

    all_files = [os.path.join(config["DATA_PATH"], filename) for filename in os.listdir(config["DATA_PATH"])]
    all_image_files = [
        instance
        for instance in all_files
        if instance.endswith(".png") or instance.endswith(".jpg") or instance.endswith(".webp")
    ]

    with open(buckets, "rb") as f:
        bucket: AspectBucket = pickle.load(f)

    vae = AutoencoderKL.from_pretrained(model, subfolder="vae")

    vae.requires_grad_(False)
    vae = vae.to(device, dtype=torch.float32)

    def vae_encode(image_filepath, latent_filepath, aspect_ratio):
        image = Image.open(image_filepath).convert("RGB")
        transforms = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize([0.5], [0.5])]
        )

        image = image.resize(aspect_ratio, resample=Image.Resampling.LANCZOS)
        # transform image to tensor
        tensor = torch.stack([transforms(image)])
        # move tensor to gpu
        tensor = tensor.to(device, memory_format=torch.contiguous_format, dtype=torch.float32).float()
        # encode latent using the vae
        latent = vae.encode(tensor).latent_dist.sample()
        latent = latent * 0.18215
        latent = latent[0]
        torch.save(latent, latent_filepath)

        del latent
        del tensor

    time = 0
    for aspect_ratio, image_list in bucket.bucket_data.items():
        time += len(image_list)

    for aspect_ratio, image_list in (pbar := progress.tqdm(bucket.bucket_data.items(), total=time)):
        for image_index in image_list:
            image_id = bucket.store.image_files[image_index][1]
            image_file_name = None
            latent_name = os.path.join(config["DATA_PATH"], image_id + ".latent")
            for ext in shared.VALID_IMAGE_EXTENSIONS:
                image_file_name = os.path.join(config["DATA_PATH"], image_id + "." + ext)
                if os.path.isfile(image_file_name):
                    break
            else:
                raise Exception(f"Extension not found for file {os.path.join(config['DATA_PATH'], image_id)}")
            vae_encode(image_file_name, latent_name, aspect_ratio)
