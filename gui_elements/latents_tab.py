import json
import pickle

import gradio as gr
import os


def make_latents(model_path, config_path, buckets_path, progress=gr.Progress()):
    try:
        print("creating latents...")
        config = json.load(open(config_path))

        from encoder import encode

        encode(model_path, buckets_path, config, progress=progress)
    except Exception as e:
        return f"Latents failed to create!\nReason: {e}"
    return "Latents created!"


def make_buckets(config_path, buckets_path, batch):
    try:
        # horribly hacky but oh well
        os.environ["SD_TRAINER_CONFIG_FILE"] = config_path
        from dataloaders.filedisk_loader import ImageStore, AspectBucket

        print("creating image store...")
        image_store = ImageStore()
        print(f"{len(image_store)} image(s).")
        print("creating aspect buckets...")
        bucket = AspectBucket(image_store, int(batch))
        print("writing aspect buckets to file...")
        with open(buckets_path, "wb") as f:
            pickle.dump(bucket, f)
    except Exception as e:
        return f"Buckets failed to create!\nReason: {e}"
    return "Buckets created!"


def load():
    model_path = gr.Textbox(lines=1, label="Model Path", value="model", interactive=True)
    config_load_path = gr.Textbox(lines=1, label="Config Path", value="train_config.json", interactive=True)
    bucket_save_path = gr.Textbox(lines=1, label="Buckets Save Path", value="buckets.pkl", interactive=True)
    buckets_button = gr.Button(value="Create Buckets")
    train_batch = gr.Number(label="Train Batch Size", value=1, interactive=True)
    latents_button = gr.Button(value="Create Latents")
    log_output = gr.components.Textbox(label="Log")

    buckets_button.click(
        fn=make_buckets, inputs=[config_load_path, bucket_save_path, train_batch], outputs=[log_output]
    )
    latents_button.click(
        fn=make_latents, inputs=[model_path, config_load_path, bucket_save_path], outputs=[log_output]
    )
