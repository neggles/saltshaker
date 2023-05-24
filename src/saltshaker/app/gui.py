import gradio as gr

from saltshaker.app.tabs import data, latents, train

with gr.Blocks(theme="gstaff/xkcd") as blocks:
    with gr.Tab("Data"):
        data.load()

    with gr.Tab("Latents"):
        latents.load()

    with gr.Tab("Train"):
        train.load()


def gui() -> None:
    blocks.queue()
    blocks.launch(
        server_name="0.0.0.0",
        server_port=7861,
    )
