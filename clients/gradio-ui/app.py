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


# ---- Generate tab ----

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


# ---- History tab ----

def load_history():
    try:
        r = requests.get(f"{ENDPOINT}/api/tasks", timeout=10)
        r.raise_for_status()
        tasks = r.json().get("tasks", [])
    except requests.RequestException:
        return [], [], "Failed to load history"

    items = []
    meta = []
    for task in reversed(tasks):
        tid = task.get("task_id", "?")
        prompt = task.get("prompt", "")[:80]
        label = f"{tid}: {prompt}"
        try:
            ir = requests.get(f"{ENDPOINT}/api/image", params={"task_id": tid, "delete": "false"}, timeout=10)
            if ir.status_code == 200:
                img = Image.open(io.BytesIO(ir.content))
                items.append((img, label))
                meta.append(task)
        except requests.RequestException:
            pass

    return items, meta, f"{len(meta)} tasks with images"


def delete_from_history(task_id):
    if not task_id:
        return [], [], "Enter a task ID"
    try:
        requests.get(f"{ENDPOINT}/api/cancel", params={"task_id": task_id}, timeout=5)
        requests.get(f"{ENDPOINT}/api/image", params={"task_id": task_id, "delete": "true"}, timeout=10)
    except requests.RequestException:
        pass
    return load_history()


def show_details(evt: gr.SelectData, meta):
    idx = evt.index
    if not meta or idx >= len(meta):
        return "No details"
    task = meta[idx]
    lines = [
        f"**Task:** `{task.get('task_id', '?')}`",
        f"**Prompt:** {task.get('prompt', '?')}",
        f"**Seed:** {task.get('seed', '?')}",
        f"**Size:** {task.get('width', '?')}×{task.get('height', '?')}",
        f"**Steps:** {task.get('steps', '?')}",
    ]
    if task.get('model_used'):
        lines.append(f"**Model:** {task['model_used']}")
    return "\n".join(lines)


# ---- UI ----

def build_ui():
    with gr.Blocks(title="Diffusion Studio") as app:
        gr.Markdown("# Diffusion Studio")
        gr.Markdown(f"Engine: `{ENDPOINT}` — [Swagger]({ENDPOINT}/swagger)")

        with gr.Tabs():
            # --- Generate tab ---
            with gr.Tab("Generate"):
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
                        output = gr.Image(label="Generated Image", type="pil", height=450, interactive=False)

                submit_btn.click(submit, [prompt, mode, init_file, edit_files, steps, width, height, seed],
                                 [task_id, status, output])
                check_btn.click(check, [task_id], [status, output])

            # --- History tab ---
            with gr.Tab("History"):
                with gr.Row():
                    load_btn = gr.Button("Refresh")
                    delete_tid = gr.Textbox(label="Task ID to delete", placeholder="Enter task ID")
                    delete_btn = gr.Button("Delete", variant="stop")
                hist_status = gr.Markdown("")
                hist_details = gr.Markdown("", visible=False)
                hist_gallery = gr.Gallery(label="Past Images", columns=3, height=500, object_fit="contain")
                hist_meta = gr.State([])

                load_btn.click(load_history, [], [hist_gallery, hist_meta, hist_status])
                delete_btn.click(delete_from_history, [delete_tid], [hist_gallery, hist_meta, hist_status])
                hist_gallery.select(show_details, [hist_meta], [hist_details]).then(
                    fn=lambda: gr.update(visible=True), outputs=[hist_details]
                )

    return app


def launch():
    port = int(os.environ.get("MFLUX_UI_PORT", "7860"))
    build_ui().launch(server_name="0.0.0.0", server_port=port, theme=gr.themes.Soft())


if __name__ == "__main__":
    launch()
