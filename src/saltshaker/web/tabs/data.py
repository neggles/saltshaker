import json
import os

import gradio as gr

import saltshaker.shared as shared
from saltshaker.data.bucket import get_buckets


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
        "BUCKETS": get_buckets(num_buckets, bucket_side_min, bucket_side_max, max_image_area**2),
        "SHUFFLE_CAPTIONS_AFTER": 0,
    }
    try:
        with open(config_path, "w") as f:
            json.dump(config, f)
        return f"Wrote config to {config_path}"
    except Exception as e:
        return f"Error: {e}"


def load():
    with gr.Row().style(equal_height=True):
        with gr.Column():
            train_data_path = gr.Textbox(
                lines=1,
                label="Dataset Directory",
            )
        with gr.Column():
            config_save_path = gr.Textbox(
                lines=1,
                label="Save bucket config to path",
                value="data/buckets.json",
                interactive=True,
            )
    with gr.Row().style(equal_height=True):
        with gr.Column(scale=1):
            max_area = gr.Slider(
                value=768,
                minimum=0,
                maximum=2048,
                label="Image Area",
                interactive=True,
                step=64,
            )
        with gr.Column(scale=1):
            generate_button = gr.Button(
                value="Generate Bucket Config",
                label="Generate Bucket Config",
                elem_id="generate_button",
                variant="primary",
            ).style(full_width=True)
    with gr.Accordion(label="Bucket settings (advanced)", open=False):
        with gr.Row().style(equal_height=True):
            with gr.Column():
                bucket_count = gr.Slider(
                    value=32,
                    minimum=0,
                    maximum=128,
                    label="Bucket Count",
                    interactive=True,
                    step=1,
                )
            with gr.Column():
                bucket_min = gr.Slider(
                    value=256,
                    minimum=0,
                    maximum=512,
                    label="Bucket Edge Min",
                    interactive=True,
                    step=64,
                )
            with gr.Column():
                bucket_max = gr.Slider(
                    value=1536,
                    minimum=0,
                    maximum=2048,
                    label="Bucket Edge Max",
                    interactive=True,
                    step=64,
                )
    with gr.Row().style(equal_height=True):
        with gr.Column():
            log_output = gr.Textbox(
                label="Log output",
                elem_id="log_output",
                lines=3,
                interactive=False,
            ).style(show_copy_button=True)
        with gr.Column():
            generated_config = gr.JSON(
                label="Generated Config",
                elem_id="generated_config",
                interactive=False,
            ).style(show_copy_button=True)

    # set event handlers
    generate_button.click(
        fn=make_config,
        inputs=[train_data_path, config_save_path, bucket_count, bucket_min, bucket_max, max_area],
        outputs=[log_output, generated_config],
    )
