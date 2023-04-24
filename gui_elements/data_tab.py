import json
import os

from create_aspect import create_aspect
import gradio as gr
import shared


def make_config(data_path, config_path, num_buckets, bucket_side_min, bucket_side_max, max_image_area):
    if not data_path:
        return "Error: No data path specified"
    if not config_path:
        return "Error: No config path specified"
    if not os.path.exists(data_path):
        return "Error: Data path does not exist"
    if not os.path.isdir(data_path):
        return "Error: Data path is not a folder"
    total_files = len(os.listdir(data_path))
    if total_files == 0:
        return "Error: No images found in data path"
    image_files = []
    caption_files = []
    for file in os.listdir(data_path):
        ext = file.split(".")[-1]
        if ext in shared.VALID_IMAGE_EXTENSIONS:
            image_files.append(file.split(".")[0])
        elif ext == "txt":
            caption_files.append(file.split(".")[0])
        else:
            return f"Error: Invalid file type {ext} for {file}"

    image_files = set(image_files)
    caption_files = set(caption_files)
    if len(image_files) != len(caption_files):
        return f"Error: Number of images and captions do not match\nOffending items: {image_files ^ caption_files}"

    config = {
        "DATA_PATH": data_path,
        "BUCKETS": create_aspect(num_buckets, bucket_side_min, bucket_side_max, max_image_area**2),
        "SHUFFLE_CAPTIONS_AFTER": 0,
    }
    try:
        with open(config_path, "w") as f:
            json.dump(config, f)
        return f"Wrote config to {config_path}"
    except Exception as e:
        return f"Error: {e}"


def load():
    train_data_path = gr.Textbox(lines=1, label="Training Data Folder Path")
    max_area = gr.Slider(
        value=768,
        minimum=0,
        maximum=2048,
        label="Max Area (input will be squared)",
        interactive=True,
        step=64,
    )
    with gr.Accordion(label="Bucket settings (advanced)", open=False):
        bucket_count = gr.Slider(
            value=32, minimum=0, maximum=128, label="Bucket Count", interactive=True, step=1
        )
        bucket_min = gr.Slider(
            value=256, minimum=0, maximum=512, label="Bucket Edge Min", interactive=True, step=64
        )
        bucket_max = gr.Slider(
            value=1536, minimum=0, maximum=2048, label="Bucket Edge Max", interactive=True, step=64
        )
    config_save_path = gr.Textbox(
        lines=1, label="Config Save Path", value="train_config.json", interactive=True
    )
    bucket_button = gr.Button(label="Create Buckets")
    log_output = gr.components.Textbox(label="Log")
    bucket_button.click(
        fn=make_config,
        inputs=[train_data_path, config_save_path, bucket_count, bucket_min, bucket_max, max_area],
        outputs=[log_output],
    )
