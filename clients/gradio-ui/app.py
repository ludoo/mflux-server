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
import time
import gradio as gr
import requests
from PIL import Image

ENDPOINT = os.environ.get("MFLUX_ENDPOINT", "http://studio:8030")


def submit(prompt, mode, init_images, steps, width, height, seed):
    """Submit a generation/edit task and return task_id + ETA."""
    try:
        steps_i = int(steps)
        width_i = int(width)
        height_i = int(height)
    except (ValueError, TypeError):
        return None, "Invalid numeric parameter", ""

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

    if mode == "img2img" and init_images:
        img = init_images[0] if isinstance(init_images, list) else init_images
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG")
        payload["init_image"] = base64.b64encode(buf.getvalue()).decode()
        payload["image_strength"] = 0.4

    if mode == "edit" and init_images:
        b64_list = []
        imgs = init_images if isinstance(init_images, list) else [init_images]
        for img in imgs:
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="PNG")
            b64_list.append(base64.b64encode(buf.getvalue()).decode())
        payload["init_images"] = b64_list

    endpoint = f"{ENDPOINT}/api/edit" if mode == "edit" else f"{ENDPOINT}/api/generate"
    try:
        r = requests.post(endpoint, json=payload, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data["task_id"], f"Queued (ETA: {data['expected_time_seconds']:.0f}s, position: {data['task_length']})", ""
    except requests.RequestException as e:
        return None, f"Error: {e}", ""


def poll(task_id, _):
    """Poll task status, return (status_text, image_or_None)."""
    if not task_id:
        return "No task", None
    try:
        r = requests.get(f"{ENDPOINT}/api/status", params={"task_id": task_id}, timeout=5)
        if r.status_code == 404:
            return "Task not found", None
        data = r.json()
        if data.get("status") == "done":
            img_r = requests.get(f"{ENDPOINT}/api/image", params={"task_id": task_id, "delete": "false"}, timeout=30)
            if img_r.status_code == 200:
                img = Image.open(io.BytesIO(img_r.content))
                return "Done!", img
            return "Done (image fetch failed)", None
        return f"Waiting... (pos: {data.get('pos', '?')}, ETA: {data.get('wait_remaining', '?')}s)", None
    except requests.RequestException as e:
        return f"Poll error: {e}", None


def build_ui():
    with gr.Blocks(title="Diffusion Studio", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Diffusion Studio")
        gr.Markdown(f"Engine: `{ENDPOINT}`  |  Swagger: [{ENDPOINT}/swagger]({ENDPOINT}/swagger)")

        with gr.Row():
            with gr.Column(scale=1):
                mode = gr.Radio(
                    ["txt2img", "img2img", "edit"],
                    value="txt2img",
                    label="Mode",
                )
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the image you want...",
                    lines=3,
                )
                init_images = gr.Gallery(
                    label="Reference Images (img2img: 1, edit: 1+)",
                    columns=3,
                    height=200,
                    type="numpy",
                )
                with gr.Row():
                    steps = gr.Slider(1, 25, value=4, step=1, label="Steps")
                    width = gr.Slider(256, 2048, value=1024, step=64, label="Width")
                with gr.Row():
                    height = gr.Slider(256, 2048, value=1024, step=64, label="Height")
                    seed = gr.Textbox(label="Seed (optional)", placeholder="auto")

                submit_btn = gr.Button("Generate", variant="primary")
                task_id = gr.Textbox(label="Task ID", visible=False)
                status_text = gr.Markdown("")

            with gr.Column(scale=1):
                output_image = gr.Image(label="Generated Image", type="pil", height=512)
                gallery = gr.Gallery(label="History", columns=2, height=400)

        # Submit flow
        submit_btn.click(
            fn=submit,
            inputs=[prompt, mode, init_images, steps, width, height, seed],
            outputs=[task_id, status_text, output_image],
        ).then(
            fn=lambda tid: gr.update(value=""),
            inputs=[],
            outputs=[output_image],
        ).then(
            fn=poll,
            inputs=[task_id, output_image],
            outputs=[status_text, output_image],
            every=3,
        )

        # When done, add to gallery
        output_image.change(
            fn=lambda img, gal: (gal or []) + [img] if img is not None else (gal or []),
            inputs=[output_image, gallery],
            outputs=[gallery],
        )

    return app


if __name__ == "__main__":
    port = int(os.environ.get("MFLUX_UI_PORT", "7860"))
    build_ui().launch(server_name="0.0.0.0", server_port=port)
