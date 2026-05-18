#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["gradio", "requests", "pillow"]
# ///
"""
Gradio UI for mflux-server image generation engine.
Run anywhere on the network — no GPU required.
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
    try:
        steps_i, width_i, height_i = int(steps), int(width), int(height)
    except (ValueError, TypeError):
        return "", "Invalid numeric parameter", None

    payload = {
        "prompt": prompt, "steps": steps_i, "width": width_i, "height": height_i,
        "format": "JPEG", "quality": 90,
    }
    if seed and seed.strip():
        payload["seed"] = seed.strip()

    if mode == "img2img" and init_file:
        payload["init_image"] = _img_to_b64(init_file)
        payload["image_strength"] = 0.4
    if mode == "edit" and edit_files:
        payload["init_images"] = [_img_to_b64(f) for f in edit_files]

    url = f"{ENDPOINT}/api/edit" if mode == "edit" else f"{ENDPOINT}/api/generate"
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        d = r.json()
        return d["task_id"], f"Submitted — ETA {d['expected_time_seconds']:.0f}s", None
    except requests.RequestException as e:
        return "", f"Error: {e}", None


def check(task_id):
    if not task_id:
        return "Enter a task ID", None
    try:
        r = requests.get(f"{ENDPOINT}/api/status", params={"task_id": task_id}, timeout=5)
        if r.status_code == 404:
            return "Task not found", None
        d = r.json()
        if d.get("status") == "done":
            ir = requests.get(f"{ENDPOINT}/api/image", params={"task_id": task_id, "delete": "false"}, timeout=30)
            if ir.status_code == 200:
                return "Done!", Image.open(io.BytesIO(ir.content))
            return "Done (fetch failed)", None
        return f"Waiting — pos {d.get('pos','?')}, ETA {d.get('wait_remaining','?')}s", None
    except requests.RequestException as e:
        return f"Error: {e}", None


def build_ui():
    with gr.Blocks(title="Diffusion Studio") as app:
        gr.Markdown("# Diffusion Studio")
        gr.Markdown(f"Engine: `{ENDPOINT}` — [Swagger]({ENDPOINT}/swagger)")

        with gr.Row():
            with gr.Column(scale=1):
                mode = gr.Radio(["txt2img", "img2img", "edit"], value="txt2img", label="Mode")
                prompt = gr.Textbox(label="Prompt", placeholder="Describe the image...", lines=3)
                init_file = gr.File(label="Init Image (img2img)", file_types=["image"], visible=False)
                edit_files = gr.File(label="Reference Images (edit)", file_types=["image"], file_count="multiple", visible=False)
                mode.change(lambda m: (gr.update(visible=(m == "img2img")), gr.update(visible=(m == "edit"))),
                           mode, [init_file, edit_files])
                with gr.Row():
                    steps = gr.Slider(1, 25, value=4, step=1, label="Steps")
                    width = gr.Slider(256, 2048, value=1024, step=64, label="Width")
                with gr.Row():
                    height = gr.Slider(256, 2048, value=1024, step=64, label="Height")
                    seed = gr.Textbox(label="Seed", placeholder="auto")
                submit_btn = gr.Button("Generate", variant="primary")

            with gr.Column(scale=1):
                task_id = gr.Textbox(label="Task ID")
                check_btn = gr.Button("Check Status")
                status = gr.Markdown("")
                output = gr.Image(label="Generated Image", type="pil", height=450)
                gallery = gr.Gallery(label="History", columns=2, height=400, object_fit="contain")

        submit_btn.click(submit, [prompt, mode, init_file, edit_files, steps, width, height, seed],
                         [task_id, status, output])
        check_btn.click(check, [task_id], [status, output])
        output.change(lambda img, gal: (gal or []) + [(img, "")] if img else (gal or []),
                      [output, gallery], [gallery])

    return app


def launch():
    port = int(os.environ.get("MFLUX_UI_PORT", "7860"))
    build_ui().launch(server_name="0.0.0.0", server_port=port, theme=gr.themes.Soft())


if __name__ == "__main__":
    launch()
