import json
import os
import subprocess
import sys

import gradio as gr


def execute(
    model_path,
    config_path,
    bucket_path,
    output_path,
    clip_penultimate,
    run_name,
    fp16,
    use_xformers,
    extended_mode_chunks,
    epochs,
    batch_size,
    train_text_encoder,
    unet_lr,
    text_encoder_lr,
    lr_scheduler,
    lr_warmup_steps,
    lr_num_cycles,
    lr_min_scale,
    lr_max_scale,
    gradient_checkpointing,
    use_ema,
    use_8bit_adam,
    adam_beta1,
    adam_beta2,
    adam_weight_decay,
    adam_epsilon,
    shuffle,
    seed,
    ucg,
    partial_dropout,
    image_log_steps,
    image_log_amount,
    save_every,
):
    config = json.load(open(config_path))
    commandline_arg = " finetune.py"
    commandline_arg += f' --model="{model_path}"'
    commandline_arg += f' --run_name="{run_name}"'
    commandline_arg += f' --dataset="{bucket_path}"'
    commandline_arg += f" --lr={unet_lr}"
    commandline_arg += f" --epochs={int(epochs)}"
    commandline_arg += f" --batch_size={int(batch_size)}"
    commandline_arg += f" --ucg={ucg}"
    commandline_arg += f" --adam_beta1={adam_beta1}"
    commandline_arg += f" --adam_beta2={adam_beta2}"
    commandline_arg += f" --adam_weight_decay={adam_weight_decay}"
    commandline_arg += f" --adam_epsilon={adam_epsilon}"
    commandline_arg += f" --seed={int(seed)}"
    commandline_arg += f' --output_path="{output_path}"'
    commandline_arg += f" --save_steps={int(save_every)}"
    commandline_arg += f" --image_log_steps={int(image_log_steps)}"
    commandline_arg += f" --image_log_amount={int(image_log_amount)}"
    commandline_arg += f" --lr_scheduler={lr_scheduler}"
    commandline_arg += f" --lr_warmup_steps={int(lr_warmup_steps)}"
    commandline_arg += f" --lr_num_cycles={int(lr_num_cycles)}"
    commandline_arg += f" --lr_min_scale={lr_min_scale}"
    commandline_arg += f" --lr_max_scale={lr_max_scale}"
    commandline_arg += f" --extended_mode_chunks={int(extended_mode_chunks)}"
    commandline_arg += f" --text_encoder_learning_rate={text_encoder_lr}"
    commandline_arg += f" --partial_dropout={partial_dropout}"
    commandline_arg += f" --shuffle={shuffle}"
    if use_8bit_adam:
        commandline_arg += f" --use_8bit_adam"
    if use_xformers:
        commandline_arg += f" --use_xformers"
    if clip_penultimate:
        commandline_arg += f" --clip_penultimate"
    if train_text_encoder:
        commandline_arg += f" --train_text_encoder"
    if gradient_checkpointing:
        commandline_arg += f" --gradient_checkpointing"
    if fp16:
        commandline_arg += f" --fp16"
    if use_ema:
        commandline_arg += f" --use_ema"

    try:
        os.environ["SD_TRAINER_CONFIG_FILE"] = config_path
        subprocess.run("accelerate launch" + commandline_arg, shell=True, check=True, env=os.environ)
    except subprocess.CalledProcessError as e:
        print(f"Command execution failed with return code {e.returncode} and error message: {e.stderr}")


def load():
    gr.Markdown("Model")
    with gr.Row():
        model_path = gr.Textbox(lines=1, label="Model Path", value="", interactive=True)
        config_path = gr.Textbox(lines=1, label="Config Path", value="train_config.json", interactive=True)
        bucket_path = gr.Textbox(lines=1, label="Bucket Path", value="buckets.pkl", interactive=True)
        output_path = gr.Textbox(lines=1, label="Output Path", value="my_finetune", interactive=True)
        save_every = gr.Number(value=1000, label="Save Every N Steps", interactive=True)
        # model_type = gr.Radio(["Checkpoint", "Diffusers"], label="Model Type", value="Checkpoint", interactive=True)
        clip_penultimate = gr.Checkbox(label="Clip Penultimate", value=False, interactive=True)
    gr.Markdown("Settings")
    with gr.Row():
        run_name = gr.Textbox(lines=1, label="Run Name", value="my_finetune", interactive=True)
        fp16 = gr.Checkbox(label="fp16", value=True, interactive=True)
        use_xformers = gr.Checkbox(label="Use xformers", value=True, interactive=True)
        extended_mode_chunks = gr.Number(value=2, label="Extended Mode Chunks")
    gr.Markdown("Hyperparams")
    with gr.Row():
        epochs = gr.Number(value=20, label="Epochs", interactive=True)
        batch_size = gr.Number(value=1, label="Batch Size", interactive=True)
        train_text_encoder = gr.Checkbox(label="Train Text Encoder", value=True, interactive=True)
    gr.Markdown("Learning Rate")
    with gr.Row():
        unet_lr = gr.Number(value=5e-7, label="Unet LR", interactive=True)
        text_encoder_lr = gr.Number(value=7e-9, label="Text Encoder LR", interactive=True)
        lr_scheduler = gr.Dropdown(
            ["linear", "cosine", "cosine_with_restarts", "polynomial"],
            label="Scheduler",
            value="cosine_with_restarts",
            interactive=True,
        )
        with gr.Accordion(label=" Scheduler Settings (advanced)", open=False):
            lr_warmup_steps = gr.Number(value=0, label="Warmup Steps", interactive=True)
            lr_num_cycles = gr.Number(value=1, label="Number of Cycles", interactive=True)
            lr_min_scale = gr.Number(value=0.0, label="Min Scale", interactive=True)
            lr_max_scale = gr.Number(value=1.0, label="Max Scale", interactive=True)
    gr.Markdown("Optimizer")
    with gr.Row():
        gradient_checkpointing = gr.Checkbox(label="Gradient Checkpointing", value=False, interactive=True)
        use_ema = gr.Checkbox(label="Use EMA", value=False, interactive=True)
        use_8bit_adam = gr.Checkbox(label="Use 8bit Adam", value=True, interactive=True)
        with gr.Accordion(label="Adam Settings (advanced)", open=False):
            adam_beta1 = gr.Number(value=0.9, label="Adam Beta 1", interactive=True)
            adam_beta2 = gr.Number(value=0.999, label="Adam Beta 2", interactive=True)
            adam_weight_decay = gr.Number(value=1e-2, label="Adam Weight Decay", interactive=True)
            adam_epsilon = gr.Number(value=1e-8, label="Adam Epsilon", interactive=True)
    gr.Markdown("Data Handling")
    with gr.Row():
        shuffle = gr.Checkbox(label="Shuffle", value=True, interactive=True)
        seed = gr.Number(value=42, label="Seed", interactive=True)
        ucg = gr.Number(value=0.1, label="Unconditional Guidance Percentage", interactive=True)
        partial_dropout = gr.Checkbox(label="Partial Dropout", value=True, interactive=True)
    gr.Markdown("Logging")
    with gr.Row():
        # report_to = gr.Dropdown(["none", "filesystem", "wandb"], label="Report To", value="none", interactive=True)
        image_log_steps = gr.Number(value=200, label="Image Log Steps", interactive=True)
        image_log_amount = gr.Number(value=4, label="Image Log Amount", interactive=True)
    start_button = gr.Button(value="Start Training", label="Start Training", interactive=True)

    start_button.click(
        execute,
        inputs=[
            model_path,
            config_path,
            bucket_path,
            output_path,
            clip_penultimate,
            run_name,
            fp16,
            use_xformers,
            extended_mode_chunks,
            epochs,
            batch_size,
            train_text_encoder,
            unet_lr,
            text_encoder_lr,
            lr_scheduler,
            lr_warmup_steps,
            lr_num_cycles,
            lr_min_scale,
            lr_max_scale,
            gradient_checkpointing,
            use_ema,
            use_8bit_adam,
            adam_beta1,
            adam_beta2,
            adam_weight_decay,
            adam_epsilon,
            shuffle,
            seed,
            ucg,
            partial_dropout,
            image_log_steps,
            image_log_amount,
            save_every,
        ],
        outputs=[],
    )
