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

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) and
[cloudflared](https://github.com/cloudflare/cloudflared/releases). The app is meant to
run on a machine you only reach over SSH, and to be used through a public Cloudflare
quick tunnel (no account or DNS setup).

```bash
uv sync               # create .venv and install locked dependencies
./start.sh --detach   # serve + tunnel in the background; survives closing SSH
./start.sh --status   # is it running? prints the https://<random>.trycloudflare.com URL
./start.sh --stop     # stop server and tunnel
```

`./start.sh` with no flag does the same in the foreground (Ctrl+C stops it), which is
handy inside `tmux`. A new trycloudflare URL can take up to a minute to start resolving.
The URL changes on every start. Logs are in `.run/` (`tunnel.log`, `cloudflared.log`).

The server binds to `127.0.0.1` only (gunicorn, one worker), on the first free port from
8090 upwards, or on `$PORT` if you set it. It refuses a busy port, so it can never publish
some other local service by mistake. If either gunicorn or cloudflared exits, the other
one is stopped too.

If cloudflared is missing, the script prints install instructions (the `.deb` or the
standalone binary from the releases page on Linux, `brew install cloudflared` on macOS).

For development without a tunnel: `./start.sh --local` (Flask dev server on
`http://127.0.0.1:${PORT:-8080}`; reach it over SSH with `ssh -L 8080:127.0.0.1:8080 host`).

## Per-visitor state

Everyone who opens the URL gets their own network. A random id in an HttpOnly cookie
selects that visitor's architecture, weights, epoch count and loss history, so students
never apply each other's gradients. The server keeps up to 500 sessions in memory,
evicting the least recently used and dropping any that sit idle for 6 hours. Restarting
the server resets all of them.

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
