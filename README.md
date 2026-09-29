# Toy Neural Network

An interactive, in-browser visualization of how a small multilayer perceptron learns.
A Flask + NumPy backend trains an MLP (ReLU hidden layers, sigmoid output, binary
cross-entropy) on a synthetic "cats vs dogs" dataset with four features. A D3 frontend
animates each forward and backward pass so you can watch activations, gradients and
weights change step by step.

## Features

- Step-by-step forward and backward pass animation
- Configurable architecture and learning rate
- Mini-batch training
- Hover over a connection to see its weight
- Click a neuron to see its calculations
- Light and dark mode

## Quickstart

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync          # create .venv and install locked dependencies
./start.sh       # or: uv run python app.py
```

Open http://127.0.0.1:8080. Set `PORT` to use a different port, for example `PORT=5000 ./start.sh`.

## Share it with a Cloudflare quick tunnel

To get a temporary public URL without an account or any DNS setup:

```bash
./tunnel.sh      # or: ./start.sh --tunnel
```

This starts the app with gunicorn on `127.0.0.1:$PORT` and runs
`cloudflared tunnel --url http://localhost:$PORT`. cloudflared prints a
`https://<random>.trycloudflare.com` URL. Press Ctrl+C to stop the tunnel and the server.

You need [cloudflared](https://github.com/cloudflare/cloudflared/releases). The script
prints install instructions if it can't find it (`brew install cloudflared` on macOS, or
the `.deb` or standalone binary from the releases page on Linux).

## Deploying

The `Procfile` (`web: gunicorn app:app --bind 0.0.0.0:$PORT --workers 1`) works on
Render, Heroku and similar platforms. They install dependencies from `requirements.txt`.

**Use exactly one worker.** The network state lives in an in-memory global in the
server process, so multiple workers or instances would each hold their own model.

`requirements.txt` is generated from `uv.lock`. Regenerate it after changing dependencies:

```bash
uv lock
uv export --no-hashes --no-dev --no-emit-project --format requirements-txt > requirements.txt
```
