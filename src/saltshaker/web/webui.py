from functools import lru_cache
from typing import Optional

import gradio as gr

from saltshaker.web.tabs import data, latents, train


@lru_cache()
def get_webui() -> gr.Blocks:
    blocks = gr.Blocks(
        title="saltshaker",
        analytics_enabled=False,
        theme=gr.themes.Base(
            primary_hue="violet",
            secondary_hue="orange",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Fira Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
            font_mono=[gr.themes.GoogleFont("Fira Code"), "ui-monospace", "Consolas", "monospace"],
        ),
    )
    with blocks:
        with gr.Tab("Data"):
            data.load()

        with gr.Tab("Latents"):
            latents.load()

        with gr.Tab("Train"):
            train.load()

    return blocks


def launch(
    host: str = "0.0.0.0",
    port: Optional[int] = None,
    share: bool = False,
) -> None:
    blocks = get_webui()
    blocks.launch(
        server_name=host,
        server_port=port,
        enable_queue=True,
        show_error=True,
        show_tips=False,
        share=share,
    )
