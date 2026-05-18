# Diffusion Studio — Gradio UI

Web UI for the [mflux-server](https://github.com/ludoo/mflux-server) image generation engine. Runs anywhere on the network — no GPU required.

## Quick Start

```bash
git clone https://github.com/ludoo/mflux-server.git
cd mflux-server
MFLUX_ENDPOINT=http://192.168.0.9:8030 uv run clients/gradio-ui/app.py
```

Opens on `http://localhost:7860`.

## Modes

| Mode | Reference images | How it works | Use for |
|------|-----------------|--------------|---------|
| **txt2img** | None | Pure text-to-image generation | Creating images from scratch |
| **img2img** | 1 image | Encodes the reference through VAE, adds noise, re-denoises with the prompt. `image_strength` (0.4) controls how much the original is preserved. Lower = more like input. | Transforming an existing image while keeping its composition |
| **edit** | 2+ images | Multi-image conditioning via Flux2KleinEdit. Reference images are concatenated as conditioning tokens — the model decides how to blend them semantically. | "Put this logo on that building", "Make this person wear these glasses" |

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `MFLUX_ENDPOINT` | `http://studio:8030` | Engine API base URL |
| `MFLUX_UI_PORT` | `7860` | UI listen port |

## Docker

```dockerfile
FROM python:3.12-slim
RUN pip install gradio requests pillow
COPY clients/gradio-ui/app.py /app/app.py
ENV MFLUX_ENDPOINT=http://studio:8030
CMD ["python", "/app/app.py"]
```

```bash
docker build -t diffusion-studio .
docker run -p 7860:7860 -e MFLUX_ENDPOINT=http://192.168.0.9:8030 diffusion-studio
```
