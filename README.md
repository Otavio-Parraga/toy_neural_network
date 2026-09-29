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
- Each visitor trains their own network, so a whole class can share one server

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
If either process exits, the script stops the other one too.

The script refuses to start if another process already listens on `$PORT`, so it can't
publish some other local service by mistake. Pick a free port with `PORT=8090 ./tunnel.sh`.

Everyone who opens the URL gets their own network. A random id in an HttpOnly cookie
selects that visitor's architecture, weights, epoch count and loss history, so students
never apply each other's gradients. The server keeps up to 500 sessions in memory,
evicting the least recently used and dropping any that sit idle for 6 hours. Restarting
the server resets all of them.

You need [cloudflared](https://github.com/cloudflare/cloudflared/releases). The script
prints install instructions if it can't find it (`brew install cloudflared` on macOS, or
the `.deb` or standalone binary from the releases page on Linux).

## Deploying

The `Procfile` (`web: gunicorn app:app --bind 0.0.0.0:$PORT --workers 1`) works on
Render, Heroku and similar platforms. They install dependencies from `requirements.txt`.

**Use exactly one worker.** Each visitor's network lives in an in-memory store inside
the server process. With several workers or instances, a visitor's requests could land on
a process that doesn't hold their network. Requests take milliseconds, so one worker
keeps up with a classroom.

`requirements.txt` is generated from `uv.lock`. Regenerate it after changing dependencies:

```bash
uv lock
uv export --no-hashes --no-dev --no-emit-project --format requirements-txt > requirements.txt
```
