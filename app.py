from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=BASE_DIR)
CORS(app)

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
    "last_loss": None,
    "last_sample_idx": 0,
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


# Initialize with defaults
net["weights"], net["biases"] = init_weights(net["architecture"])


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(BASE_DIR, filename)


@app.route("/api/init", methods=["POST"])
def api_init():
    data = request.get_json() or {}
    arch = data.get("architecture", [4, 4, 1])
    arch = [4] + [max(1, int(h)) for h in arch[1:-1]] + [1]  # keep input=4, output=1
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
    data = request.get_json() or {}
    idx = int(data.get("sample_idx", 0)) % len(X_train)
    x = X_train[idx]
    y = float(y_train[idx])

    result = forward_backward(x, y, net["weights"], net["biases"])

    # Store gradients so /api/update can use them
    net["grad_w"]         = result["grad_w"]
    net["grad_b"]         = result["grad_b"]
    net["last_loss"]      = result["loss"]
    net["last_sample_idx"]= idx

    return jsonify({
        **result,
        "architecture":   net["architecture"],
        "weights":        [w.tolist() for w in net["weights"]],
        "biases":         [b.tolist() for b in net["biases"]],
        "features_raw":   X_raw[idx].tolist(),
        "features_norm":  x.tolist(),
        "true_label":     int(y_train[idx]),
        "sample_idx":     idx,
        "epoch":          net["epoch"],
        "loss_history":   net["loss_history"],
        "feature_names":  FEATURE_NAMES,
    })


@app.route("/api/update", methods=["POST"])
def api_update():
    if net["grad_w"] is None:
        return jsonify({"error": "Run compute first"}), 400

    data = request.get_json() or {}
    lr   = float(data.get("lr", 0.1))

    weights_old = [w.tolist() for w in net["weights"]]
    biases_old  = [b.tolist() for b in net["biases"]]
    grad_w = [np.array(g) for g in net["grad_w"]]
    grad_b = [np.array(g) for g in net["grad_b"]]

    deltas_w = [(-lr * dW).tolist() for dW in grad_w]

    for i in range(len(net["weights"])):
        net["weights"][i] -= lr * grad_w[i]
        net["biases"][i]  -= lr * grad_b[i]

    net["epoch"] += 1
    net["loss_history"].append(float(net["last_loss"]))

    # Recompute loss on same sample with new weights to show improvement
    idx = net["last_sample_idx"]
    new_result = forward_backward(X_train[idx], float(y_train[idx]),
                                  net["weights"], net["biases"])

    return jsonify({
        "weights_old":  weights_old,
        "weights_new":  [w.tolist() for w in net["weights"]],
        "biases_old":   biases_old,
        "biases_new":   [b.tolist() for b in net["biases"]],
        "deltas_w":     deltas_w,
        "loss_before":  net["last_loss"],
        "loss_after":   new_result["loss"],
        "epoch":        net["epoch"],
        "loss_history": net["loss_history"],
    })


if __name__ == "__main__":
    print("Open: http://127.0.0.1:8080")
    app.run(debug=True, host="127.0.0.1", port=8080)
