#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["gradio", "requests", "pillow"]
# ///
"""
Gradio UI for mflux-server image generation engine.
Run anywhere on the network — no GPU required.
Talks to the engine at MFLUX_ENDPOINT (default http://studio:8030).
"""

import base64
import io
import os
import gradio as gr
import requests
from PIL import Image

ENDPOINT = os.environ.get("MFLUX_ENDPOINT", "http://studio:8030")


def _img_to_b64(path):
    with Image.open(path) as img:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()


def submit(prompt, mode, init_file, edit_files, steps, width, height, seed):
    """Submit a generation/edit task."""
    try:
        steps_i = int(steps)
        width_i = int(width)
        height_i = int(height)
    except (ValueError, TypeError):
        return None, None, "Invalid numeric parameter"

    payload = {
        "prompt": prompt,
        "steps": steps_i,
        "width": width_i,
        "height": height_i,
        "format": "JPEG",
        "quality": 90,
    }
    if seed and seed.strip():
        payload["seed"] = seed.strip()

    # img2img: single file
    if mode == "img2img" and init_file:
        payload["init_image"] = _img_to_b64(init_file)
        payload["image_strength"] = 0.4

    # edit: multiple files
    if mode == "edit" and edit_files:
        payload["init_images"] = [_img_to_b64(f) for f in edit_files]

    endpoint = f"{ENDPOINT}/api/edit" if mode == "edit" else f"{ENDPOINT}/api/generate"
    try:
        r = requests.post(endpoint, json=payload, timeout=10)
        r.raise_for_status()
        data = r.json()
        return (data["task_id"],
                f"Queued — ETA {data['expected_time_seconds']:.0f}s, position {data['task_length']}",
                None)
    except requests.RequestException as e:
        return None, f"Error: {e}", None


def poll(task_id, current_img):
    """Poll task status."""
    if not task_id:
        return "No task", current_img
    try:
        r = requests.get(f"{ENDPOINT}/api/status", params={"task_id": task_id}, timeout=5)
        if r.status_code == 404:
            return "Task not found", current_img
        data = r.json()
        if data.get("status") == "done":
            img_r = requests.get(f"{ENDPOINT}/api/image",
                                params={"task_id": task_id, "delete": "false"}, timeout=30)
            if img_r.status_code == 200:
                img = Image.open(io.BytesIO(img_r.content))
                return "Done!", img
            return "Done (image fetch failed)", current_img
        return f"Waiting... pos {data.get('pos', '?')}, ETA {data.get('wait_remaining', '?')}s", current_img
    except requests.RequestException as e:
        return f"Poll error: {e}", current_img


def add_to_gallery(img, gallery):
    if img is None:
        return gallery or []
    return (gallery or []) + [(img, "")]
    

def build_ui():
    with gr.Blocks(title="Diffusion Studio") as app:
        gr.Markdown("# Diffusion Studio")
        gr.Markdown(f"Engine: `{ENDPOINT}`  —  [Swagger]({ENDPOINT}/swagger)")

        with gr.Row():
            with gr.Column(scale=1):
                mode = gr.Radio(["txt2img", "img2img", "edit"], value="txt2img", label="Mode")
                prompt = gr.Textbox(label="Prompt", placeholder="Describe the image...", lines=3)

                init_file = gr.File(label="Init Image (img2img)", file_types=["image"], visible=False)
                edit_files = gr.File(label="Reference Images (edit)", file_types=["image"], file_count="multiple", visible=False)
                
                def toggle_mode(m):
                    return (gr.update(visible=(m == "img2img")),
                            gr.update(visible=(m == "edit")))
                mode.change(toggle_mode, mode, [init_file, edit_files])

                with gr.Row():
                    steps = gr.Slider(1, 25, value=4, step=1, label="Steps")
                    width = gr.Slider(256, 2048, value=1024, step=64, label="Width")
                with gr.Row():
                    height = gr.Slider(256, 2048, value=1024, step=64, label="Height")
                    seed = gr.Textbox(label="Seed", placeholder="auto")

                submit_btn = gr.Button("Generate", variant="primary")
                task_id = gr.Textbox(visible=False)
                status_text = gr.Markdown("")

            with gr.Column(scale=1):
                output_image = gr.Image(label="Generated Image", type="pil", height=450)
                gallery = gr.Gallery(label="History", columns=2, height=400, object_fit="contain")

        submit_btn.click(
            fn=submit,
            inputs=[prompt, mode, init_file, edit_files, steps, width, height, seed],
            outputs=[task_id, status_text, output_image],
        )

        gr.Timer(3).tick(
            fn=poll,
            inputs=[task_id, output_image],
            outputs=[status_text, output_image],
        )

        output_image.change(fn=add_to_gallery, inputs=[output_image, gallery], outputs=[gallery])

    return app


def launch():
    port = int(os.environ.get("MFLUX_UI_PORT", "7860"))
    build_ui().launch(server_name="0.0.0.0", server_port=port, theme=gr.themes.Soft())


if __name__ == "__main__":
    launch()
