import gradio as gr
import gui_elements.data_tab as data_tab
import gui_elements.latents_tab as latents_tab
import gui_elements.train_tab as train_tab

with gr.Blocks(theme='gstaff/xkcd') as demo:
    with gr.Tab("Data"):
        data_tab.load()

    with gr.Tab("Latents"):
        latents_tab.load()

    with gr.Tab("Train"):
        train_tab.load()


demo.queue().launch()
