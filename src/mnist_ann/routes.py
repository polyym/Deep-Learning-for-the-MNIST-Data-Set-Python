"""HTTP request handlers mounted as a Flask Blueprint.

Keeps every ``/api/*`` endpoint (plus the root static-file handler) in one
place so :mod:`mnist_ann.app` stays focused on app construction.
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from io import BytesIO

from flask import Blueprint, current_app, jsonify, request, send_from_directory
from PIL import Image, UnidentifiedImageError

from .backend import GPU_AVAILABLE
from .data import get_data_path, load_mnist_data
from .extensions import limiter
from .network import NeuralNetwork
from .preprocessing import preprocess_drawing
from .state import training_state
from .validation import (
    ValidationError,
    require_json,
    validate_choice,
    validate_digit,
    validate_float,
    validate_int,
)

logger = logging.getLogger(__name__)

bp = Blueprint("api", __name__)


# ---------------------------------------------------------------------------
# Static
# ---------------------------------------------------------------------------
@bp.route("/")
def index():
    """Serve the single-page React UI."""
    return send_from_directory(current_app.static_folder, "index.html")


# ---------------------------------------------------------------------------
# Health & status
# ---------------------------------------------------------------------------
@bp.route("/api/health", methods=["GET"])
@limiter.exempt
def health_check():
    """Lightweight liveness probe with backend info."""
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "is_training": training_state.is_training,
        "has_model": training_state.network is not None,
        "has_results": training_state.results is not None,
        "gpu_available": GPU_AVAILABLE,
    })


@bp.route("/api/status", methods=["GET"])
@limiter.limit("600 per minute")
def get_status():
    """Return the current training status and a recent progress tail.

    The UI polls this every 500 ms during training (~120/min); the cap is
    5x that headroom so multi-tab use is fine, but caps a single IP from
    spamming status checks.
    """
    return jsonify({
        "is_training": training_state.is_training,
        "progress": training_state.get_progress(20),
        "has_results": training_state.results is not None,
    })


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@bp.route("/api/train", methods=["POST"])
@limiter.limit("5 per minute")
@require_json
def train_network():
    """Kick off a background training run with validated hyperparameters."""
    data = request.json

    try:
        n_epochs = validate_int(
            data.get("epochs"), "epochs",
            min_val=1, max_val=10000, default=100,
        )
        learning_rate = validate_float(
            data.get("learningRate"), "learningRate",
            min_val=0.0001, max_val=10.0, default=0.01,
        )
        batch_size = validate_int(
            data.get("batchSize"), "batchSize",
            min_val=1, max_val=4096, default=32,
        )
        use_small_data = bool(data.get("useSmallData", True))
        use_gpu = bool(data.get("useGpu", True))
        backprop_method = validate_choice(
            data.get("backpropMethod"), "backpropMethod",
            ["cb", "uhb"], default="cb",
        )
        hidden_u = validate_int(
            data.get("hiddenU"), "hiddenU",
            min_val=1, max_val=1024, default=64,
        )
        hidden_v = validate_int(
            data.get("hiddenV"), "hiddenV",
            min_val=1, max_val=1024, default=32,
        )
        hidden_w = validate_int(
            data.get("hiddenW"), "hiddenW",
            min_val=1, max_val=1024, default=16,
        )
        digit_a = validate_digit(data.get("digitA"), "digitA", default=0)
        digit_b = validate_digit(data.get("digitB"), "digitB", default=1)
        digit_c = validate_digit(data.get("digitC"), "digitC", default=2)
        digit_d = validate_digit(data.get("digitD"), "digitD", default=3)

        digits = [digit_a, digit_b, digit_c, digit_d]
        if len(set(digits)) != len(digits):
            raise ValidationError("All four digits must be unique")

    except ValidationError as e:
        logger.warning("Training request validation failed: %s", e)
        return jsonify({"error": str(e)}), 400

    # Atomically claim the training slot; rejects concurrent requests
    # without racing on the check-then-set. Keeps the previously trained
    # network (if any) so predict/sample_images still work until a
    # successful run replaces it.
    if not training_state.start_if_idle():
        logger.warning("Training request rejected: training already in progress")
        return jsonify({"error": "Training already in progress"}), 400

    gpu_status = "GPU" if (use_gpu and GPU_AVAILABLE) else "CPU"
    logger.info(
        "Starting training [%s]: epochs=%d, lr=%s, batch_size=%d, method=%s, "
        "hidden=(%d,%d,%d), digits=%s",
        gpu_status, n_epochs, learning_rate, batch_size, backprop_method,
        hidden_u, hidden_v, hidden_w, digits,
    )

    def progress_callback(info):
        """Translate a training-loop info dict into the JSON shape the UI polls."""
        payload = {
            "epoch": info["epoch"],
            "totalEpochs": info["total_epochs"],
            "loss": float(info["loss"]),
            "accuracy": float(info["accuracy"]),
            "timestamp": time.time(),
        }
        if "samples_done" in info:
            payload["samples_done"] = info["samples_done"]
            payload["total_samples"] = info["total_samples"]
        training_state.add_progress(payload)

    def train_thread():
        """Background worker: build net, load data, train, evaluate, publish results.

        Publishes ``training_state.network`` and ``training_state.results`` only
        on full success; cancellation or any exception leaves the prior model
        (if any) untouched. Always clears ``is_training`` via ``finally``.
        """
        try:
            # Kept local until training + evaluation succeed so a partial
            # run can't be exposed through /api/predict.
            nn = NeuralNetwork(
                hidden_layers=(hidden_u, hidden_v, hidden_w),
                learning_rate=learning_rate,
                digits_to_classify=(digit_a, digit_b, digit_c, digit_d),
                use_gpu=use_gpu,
            )

            train_path = get_data_path(use_small_data, train=True)
            X_train, labels_train = load_mnist_data(train_path)
            Y_train = nn.one_hot_encode(labels_train)
            logger.info("Loaded training data: %d samples", X_train.shape[1])

            history = nn.train(
                X_train, Y_train,
                n_epochs=n_epochs,
                batch_size=batch_size,
                backprop_method=backprop_method,
                progress_callback=progress_callback,
                should_cancel=training_state.should_cancel,
            )

            if history.get("cancelled"):
                logger.info("Training cancelled by user")
                training_state.add_progress({
                    "cancelled": True,
                    "timestamp": time.time(),
                    "message": "Training cancelled",
                })
                return

            if history.get("diverged"):
                msg = (
                    "Training diverged (loss became NaN/inf). Try a smaller "
                    "learning rate; UHB is especially sensitive and the paper "
                    "used lr=0.001."
                )
                logger.warning(msg)
                training_state.add_progress({
                    "error": msg,
                    "timestamp": time.time(),
                })
                return

            train_results = nn.evaluate(X_train, Y_train)

            test_path = get_data_path(use_small_data, train=False)
            X_test, labels_test = load_mnist_data(test_path)
            Y_test = nn.one_hot_encode(labels_test)
            test_results = nn.evaluate(X_test, Y_test)

            logger.info(
                "Training complete: train_acc=%.4f, test_acc=%.4f",
                train_results["accuracy"], test_results["accuracy"],
            )

            # Publish the model and results only on full success.
            training_state.network = nn
            training_state.results = {
                "training": train_results,
                "testing": test_results,
                "history": history,
                "config": {
                    "epochs": n_epochs,
                    "learningRate": learning_rate,
                    "batchSize": batch_size,
                    "backpropMethod": backprop_method,
                    "hiddenLayers": [hidden_u, hidden_v, hidden_w],
                    "digits": digits,
                    "useSmallData": use_small_data,
                },
            }

        except FileNotFoundError as e:
            error_msg = f"Dataset error: {e}"
            logger.error(error_msg)
            training_state.add_progress({
                "error": error_msg,
                "timestamp": time.time(),
            })
        except Exception as e:
            logger.exception("Training failed")
            training_state.add_progress({
                "error": str(e),
                "timestamp": time.time(),
            })
        finally:
            training_state.is_training = False

    thread = threading.Thread(target=train_thread, daemon=True)
    thread.start()

    return jsonify({"message": "Training started"})


@bp.route("/api/cancel", methods=["POST"])
@limiter.limit("30 per minute")
def cancel_training():
    """Request cancellation of an in-flight training run."""
    if training_state.request_cancel():
        logger.info("Cancellation requested")
        return jsonify({"message": "Cancellation requested"})
    return jsonify({"error": "No training in progress"}), 400


@bp.route("/api/results", methods=["GET"])
@limiter.exempt
def get_results():
    """Return the latest published training + testing results."""
    results = training_state.results
    if results is None:
        return jsonify({"error": "No results available"}), 404
    return jsonify(results)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
@bp.route("/api/predict", methods=["POST"])
@limiter.limit("30 per minute")
@require_json
def predict():
    """Classify a hand-drawn digit supplied as a base64 PNG."""
    nn = training_state.network
    if nn is None:
        return jsonify({"error": "No trained model available"}), 400

    data = request.json
    image_data = data.get("image")

    if not image_data:
        return jsonify({"error": "No image provided"}), 400

    if not isinstance(image_data, str):
        return jsonify({"error": "Image must be a base64 string"}), 400

    # Rough ceiling; real limit is MAX_CONTENT_LENGTH at the Flask level.
    if len(image_data) > 1.5 * 1024 * 1024:
        return jsonify({"error": "Image too large"}), 400

    try:
        if "base64," in image_data:
            # ``maxsplit=1`` so a stray "base64," later in the payload
            # (astronomically unlikely -- comma isn't in the base64
            # alphabet -- but defensive) can't drop the real payload.
            image_data = image_data.split("base64,", 1)[1]

        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))

        # MNIST-style preprocessing: no polarity inversion (canvas already
        # has bright digit on dark bg), plus bbox crop, 20x20 scale, and
        # centre-of-mass centring on a 28x28 canvas.
        processed = preprocess_drawing(image)
        X = processed.flatten()

        pred_class, probabilities = nn.predict(X)
        predicted_digit = nn.get_digit_label(pred_class[0])

        logger.debug("Prediction made: %s", predicted_digit)

        return jsonify({
            "prediction": predicted_digit,
            "class_index": int(pred_class[0]),
            "probabilities": {
                label: float(probabilities[i, 0])
                for i, label in enumerate(nn.class_labels)
            },
        })
    except base64.binascii.Error:
        logger.warning("Invalid base64 image data received")
        return jsonify({"error": "Invalid base64 image data"}), 400
    except (UnidentifiedImageError, ValueError, OSError):
        # PIL raises UnidentifiedImageError for non-image data, ValueError for
        # malformed images, and OSError for truncated streams. Predictable
        # client-side errors, not server bugs -> 400, not 500.
        logger.warning("Invalid image payload")
        return jsonify({"error": "Invalid image payload"}), 400
    except Exception:
        logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed"}), 500


@bp.route("/api/sample_images", methods=["GET"])
def get_sample_images():
    """Return a handful of sample test images with predictions.

    Uses whichever test set matches the dataset the current model was
    trained on (falls back to the small set if the config isn't stashed).
    """
    nn = training_state.network
    if nn is None:
        return jsonify({"error": "No trained model available"}), 400

    results = training_state.results or {}
    use_small = bool(results.get("config", {}).get("useSmallData", True))

    try:
        test_path = get_data_path(use_small, train=False)
        X_test, labels_test = load_mnist_data(test_path)

        n_samples = min(10, X_test.shape[1])
        # One batched forward pass instead of ten single-sample ones.
        pred_classes, probs = nn.predict(X_test[:, :n_samples])

        samples = [
            {
                "image": (X_test[:, i] * 255).astype(int).tolist(),
                "true_label": int(labels_test[i]),
                "predicted": nn.get_digit_label(int(pred_classes[i])),
                "probabilities": probs[:, i].tolist(),
            }
            for i in range(n_samples)
        ]

        return jsonify({"samples": samples})
    except FileNotFoundError as e:
        logger.error("Sample images failed: %s", e)
        return jsonify({"error": "Test dataset not found"}), 500
    except (OSError, ValueError):
        # Corrupted CSV / disk errors; log then surface a generic 500.
        logger.exception("Failed to read sample test data")
        return jsonify({"error": "Failed to load sample images"}), 500
