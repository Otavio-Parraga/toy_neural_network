from flask import Flask, abort, jsonify, request, send_from_directory
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS
import json
import math
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_FILES = {"index.html", "app.js", "style.css"}

# Limits mirrored from the UI (hidden layer inputs are 1..16, at most 5 layers).
MIN_HIDDEN_SIZE, MAX_HIDDEN_SIZE = 1, 16
MIN_HIDDEN_LAYERS, MAX_HIDDEN_LAYERS = 1, 5
MAX_LR = 10.0


def _finite_or_none(obj):
    """Recursively replace NaN/Infinity floats with None (valid JSON null)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite_or_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite_or_none(v) for v in obj]
    return obj


class StrictJSONProvider(DefaultJSONProvider):
    """Never emit the non-standard NaN / Infinity tokens in responses."""

    def dumps(self, obj, **kwargs):
        kwargs.setdefault("default", self.default)
        kwargs.setdefault("ensure_ascii", self.ensure_ascii)
        kwargs.setdefault("sort_keys", self.sort_keys)
        kwargs["allow_nan"] = False
        try:
            return json.dumps(obj, **kwargs)
        except ValueError:
            return json.dumps(_finite_or_none(obj), **kwargs)


app = Flask(__name__, static_folder=BASE_DIR)
app.json = StrictJSONProvider(app)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024   # API bodies are tiny
CORS(app)


class BadRequest(ValueError):
    """Invalid client input; turned into a 400 {"error": ...} response."""


@app.errorhandler(BadRequest)
def handle_bad_request(e):
    return jsonify({"error": str(e)}), 400


def json_body():
    if not request.get_data(cache=True):
        return {}
    data = request.get_json(force=True, silent=True)
    if data is None:
        raise BadRequest("Request body must be valid JSON")
    if not isinstance(data, dict):
        raise BadRequest("Request body must be a JSON object")
    return data


def parse_int(value, name):
    """Accept an int, an integral finite float, or a decimal integer string."""
    if isinstance(value, bool):
        raise BadRequest(f"{name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    if isinstance(value, str) and len(value) <= 12:
        try:
            return int(value.strip())
        except ValueError:
            pass
    raise BadRequest(f"{name} must be an integer")


def parse_lr(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise BadRequest("lr must be a number")
    try:
        lr = float(value)
    except (ValueError, OverflowError):
        raise BadRequest("lr must be a number")
    if not math.isfinite(lr) or not (0.0 < lr <= MAX_LR):
        raise BadRequest(f"lr must be a finite number with 0 < lr <= {MAX_LR:g}")
    return lr


def parse_architecture(value):
    """Validate a [4, h1, ..., hk, 1] list; input/output sizes are forced to 4/1."""
    if not isinstance(value, list):
        raise BadRequest("architecture must be a list of integers")
    hidden = value[1:-1]
    if len(hidden) < MIN_HIDDEN_LAYERS:
        raise BadRequest(f"architecture needs at least {MIN_HIDDEN_LAYERS} hidden layer")
    hidden = hidden[:MAX_HIDDEN_LAYERS]
    sizes = [max(MIN_HIDDEN_SIZE, min(MAX_HIDDEN_SIZE, parse_int(h, "hidden layer size")))
             for h in hidden]
    return [4] + sizes + [1]

# ── Synthetic cats vs dogs data ───────────────────────────────────────────────
FEATURE_NAMES = ["Weight (kg)", "Ear Pointiness", "Meow/Bark Ratio", "Agility Score"]


def generate_data(n=100, seed=42):
    np.random.seed(seed)
    n2 = n // 2
    cats = np.column_stack([
        np.clip(np.random.normal(4.2, 0.8, n2), 1.0, 10.0),
        np.clip(np.random.normal(0.82, 0.08, n2), 0.0, 1.0),
        np.clip(np.random.normal(0.88, 0.08, n2), 0.0, 1.0),
        np.clip(np.random.normal(0.78, 0.10, n2), 0.0, 1.0),
    ])
    dogs = np.column_stack([
        np.clip(np.random.normal(18.0, 8.0, n2), 2.0, 60.0),
        np.clip(np.random.normal(0.28, 0.12, n2), 0.0, 1.0),
        np.clip(np.random.normal(0.12, 0.08, n2), 0.0, 1.0),
        np.clip(np.random.normal(0.48, 0.12, n2), 0.0, 1.0),
    ])
    X_raw = np.vstack([cats, dogs])
    y = np.array([0.0] * n2 + [1.0] * n2)
    X_mean, X_std = X_raw.mean(0), X_raw.std(0)
    X_norm = (X_raw - X_mean) / X_std
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    return X_norm[idx], y[idx], X_raw[idx]


X_train, y_train, X_raw = generate_data()

# ── Network state (server-side) ───────────────────────────────────────────────
net = {
    "architecture": [4, 4, 1],
    "weights": [],
    "biases": [],
    "epoch": 0,
    "loss_history": [],
    # Last forward/backward values for current sample
    "grad_w": None,
    "grad_b": None,
    "last_loss": None,            # mean loss over the last computed batch
    "last_sample_idx": 0,         # focus (first) sample of the last batch
    "last_batch_indices": [0],    # all sample indices of the last batch
}


def relu(x):
    return np.maximum(0.0, x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def init_weights(architecture, seed=42):
    np.random.seed(seed)
    weights, biases = [], []
    for i in range(len(architecture) - 1):
        fan_in, fan_out = architecture[i], architecture[i + 1]
        W = np.random.randn(fan_out, fan_in) * np.sqrt(2.0 / fan_in)
        b = np.zeros(fan_out)
        weights.append(W)
        biases.append(b)
    return weights, biases


def forward_backward(x, y_true, weights, biases):
    """Full forward + backward for ONE sample. Returns every intermediate value."""
    n_w = len(weights)

    # ── forward ──────────────────────────────────────────────────────
    activations = [x.copy()]          # activations[i] = output of layer i
    pre_activations = []              # pre_activations[i] = z before activation of layer i+1

    h = x.copy()
    for i, (W, b) in enumerate(zip(weights, biases)):
        z = W @ h + b
        pre_activations.append(z.copy())
        h = sigmoid(z) if i == n_w - 1 else relu(z)
        activations.append(h.copy())

    y_pred = float(activations[-1][0])
    eps = 1e-8
    loss = -(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))

    # ── backward ─────────────────────────────────────────────────────
    # grad_z[i]          = dL/dz for layer i+1 (weight matrix i)
    # grad_w[i]          = dL/dW for weight matrix i
    # grad_b[i]          = dL/db for bias vector i
    # grad_a_hidden[i]   = dL/da for hidden activation layer i+1 (only hidden layers)
    grad_z, grad_w, grad_b = [None] * n_w, [None] * n_w, [None] * n_w
    grad_a_hidden = [None] * (n_w - 1)   # one per hidden layer

    # output: BCE + sigmoid combined → dL/dz_out = ŷ − y
    dz = activations[-1] - np.array([y_true])

    for i in reversed(range(n_w)):
        dW = np.outer(dz, activations[i])
        db = dz.copy()
        grad_z[i] = dz.copy()
        grad_w[i] = dW
        grad_b[i] = db
        if i > 0:
            da = weights[i].T @ dz
            grad_a_hidden[i - 1] = da.copy()
            dz = da * (pre_activations[i - 1] > 0).astype(float)  # ReLU derivative

    return {
        "activations":    [a.tolist() for a in activations],
        "pre_activations":[z.tolist() for z in pre_activations],
        "loss":           float(loss),
        "prediction":     int(y_pred > 0.5),
        "probability":    y_pred,
        "grad_z":         [g.tolist() for g in grad_z],
        "grad_w":         [g.tolist() for g in grad_w],
        "grad_b":         [g.tolist() for g in grad_b],
        "grad_a_hidden":  [g.tolist() if g is not None else None for g in grad_a_hidden],
    }


def batch_indices(start, batch_size):
    """Indices [start, start+1, ..., start+B-1] modulo n_samples, B clamped to [1, n]."""
    n = len(X_train)
    B = max(1, min(n, parse_int(batch_size, "batch_size")))
    start = parse_int(start, "sample_idx") % n
    return [(start + k) % n for k in range(B)]


def run_batch(indices, weights, biases):
    """forward_backward for every sample in the batch.

    Returns (per-sample results, batch-averaged grad_w, batch-averaged grad_b, mean loss).
    Averaged gradients: dL/dW = (1/B) * sum_i dL_i/dW  (L = mean BCE over the batch).
    """
    results = [forward_backward(X_train[i], float(y_train[i]), weights, biases)
               for i in indices]
    B = len(results)
    n_w = len(weights)
    avg_w = [np.mean([np.array(r["grad_w"][l]) for r in results], axis=0) for l in range(n_w)]
    avg_b = [np.mean([np.array(r["grad_b"][l]) for r in results], axis=0) for l in range(n_w)]
    mean_loss = float(sum(r["loss"] for r in results) / B)
    return results, avg_w, avg_b, mean_loss


# Initialize with defaults
net["weights"], net["biases"] = init_weights(net["architecture"])


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    # Only serve the frontend assets; never expose other project files
    # (pyproject.toml, .venv, ...) — the app may be public via a tunnel.
    if filename not in STATIC_FILES:
        abort(404)
    return send_from_directory(BASE_DIR, filename)


@app.route("/api/init", methods=["POST"])
def api_init():
    data = json_body()
    arch = parse_architecture(data.get("architecture", [4, 4, 1]))
    net["architecture"] = arch
    net["weights"], net["biases"] = init_weights(arch)
    net["epoch"] = 0
    net["loss_history"] = []
    net["grad_w"] = net["grad_b"] = net["last_loss"] = None
    return jsonify({
        "architecture": arch,
        "weights": [w.tolist() for w in net["weights"]],
        "biases":  [b.tolist() for b in net["biases"]],
        "n_samples": len(y_train),
        "y_labels": y_train.tolist(),
        "X_raw": X_raw.tolist(),
    })


@app.route("/api/compute", methods=["POST"])
def api_compute():
    data = json_body()
    indices = batch_indices(data.get("sample_idx", 0), data.get("batch_size", 1))
    idx = indices[0]                       # focus sample = first of the batch

    results, avg_w, avg_b, mean_loss = run_batch(indices, net["weights"], net["biases"])
    result = results[0]

    # Store batch-averaged gradients so /api/update can apply them
    net["grad_w"]             = [g.tolist() for g in avg_w]
    net["grad_b"]             = [g.tolist() for g in avg_b]
    net["last_loss"]          = mean_loss
    net["last_sample_idx"]    = idx
    net["last_batch_indices"] = indices

    samples = [{
        **r,
        "features_raw":  X_raw[i].tolist(),
        "features_norm": X_train[i].tolist(),
        "true_label":    int(y_train[i]),
        "sample_idx":    i,
    } for i, r in zip(indices, results)]

    return jsonify({
        **result,
        # grad_w / grad_b are the batch-AVERAGED gradients (what the update applies);
        # the focus sample's own gradients are kept as sample_grad_w / sample_grad_b.
        "grad_w":         net["grad_w"],
        "grad_b":         net["grad_b"],
        "sample_grad_w":  result["grad_w"],
        "sample_grad_b":  result["grad_b"],
        "mean_loss":      mean_loss,
        "batch": {
            "size":          len(indices),
            "indices":       indices,
            "true_labels":   [int(y_train[i]) for i in indices],
            "probabilities": [r["probability"] for r in results],
            "predictions":   [r["prediction"] for r in results],
            "losses":        [r["loss"] for r in results],
            "mean_loss":     mean_loss,
            "samples":       samples,
        },
        "architecture":   net["architecture"],
        "weights":        [w.tolist() for w in net["weights"]],
        "biases":         [b.tolist() for b in net["biases"]],
        "features_raw":   X_raw[idx].tolist(),
        "features_norm":  X_train[idx].tolist(),
        "true_label":     int(y_train[idx]),
        "sample_idx":     idx,
        "epoch":          net["epoch"],
        "loss_history":   net["loss_history"],
        "feature_names":  FEATURE_NAMES,
    })


@app.route("/api/update", methods=["POST"])
def api_update():
    data = json_body()
    lr   = parse_lr(data.get("lr", 0.1))

    if net["grad_w"] is None:
        return jsonify({"error": "Run compute first"}), 400

    weights_old = [w.tolist() for w in net["weights"]]
    biases_old  = [b.tolist() for b in net["biases"]]
    grad_w = [np.array(g) for g in net["grad_w"]]   # batch-averaged
    grad_b = [np.array(g) for g in net["grad_b"]]

    deltas_w = [(-lr * dW).tolist() for dW in grad_w]

    new_w = [W - lr * dW for W, dW in zip(net["weights"], grad_w)]
    new_b = [b - lr * db for b, db in zip(net["biases"], grad_b)]
    if not all(np.all(np.isfinite(a)) for a in new_w + new_b):
        return jsonify({"error": "Update would make weights non-finite; "
                                 "try a smaller learning rate"}), 400
    net["weights"], net["biases"] = new_w, new_b

    loss_before = net["last_loss"]
    net["epoch"] += 1
    net["loss_history"].append(float(loss_before))   # mean batch loss
    # These gradients are now spent: a repeated update needs a fresh compute.
    net["grad_w"] = net["grad_b"] = net["last_loss"] = None

    # Recompute loss on the same batch with new weights to show improvement
    indices = net["last_batch_indices"]
    new_results, _, _, mean_after = run_batch(indices, net["weights"], net["biases"])

    return jsonify({
        "weights_old":  weights_old,
        "weights_new":  [w.tolist() for w in net["weights"]],
        "biases_old":   biases_old,
        "biases_new":   [b.tolist() for b in net["biases"]],
        "deltas_w":     deltas_w,
        "loss_before":  loss_before,
        "loss_after":   mean_after,
        "batch_losses_after": [r["loss"] for r in new_results],
        "epoch":        net["epoch"],
        "loss_history": net["loss_history"],
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host="0.0.0.0", port=port)
