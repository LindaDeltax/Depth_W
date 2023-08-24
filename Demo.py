import gradio as gr
import torch

from .depth_prediction import create_demo as create_depth_pred_demo
from .trimesh_3d import create_demo as create_im_to_3d_demo


css = """
#img-display-container {
    max-height: 50vh;
    }
#img-display-input {
    max-height: 40vh;
    }
#img-display-output {
    max-height: 40vh;
    }
    
"""
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('isl-org/ZoeDepth', "ZoeD_N", pretrained=True).to(DEVICE).eval()

title = "# Depth Map Prediction"
description = "Demo for Depth Map and 3D Point Cloud from single image by Combining Relative and Metric Depth."

with gr.Blocks(css=css) as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    with gr.Tab("Depth Prediction"):
        create_depth_pred_demo(model)
    with gr.Tab("Image to 3D"):
        create_im_to_3d_demo(model)

if __name__ == '__main__':
    demo.queue().launch()
