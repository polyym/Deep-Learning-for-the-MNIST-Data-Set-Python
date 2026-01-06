"""
Flask Backend for MNIST Neural Network Web Application

Production-ready implementation with:
- Environment variable configuration
- Thread-safe state management
- Input validation
- Proper error logging
- Health check endpoint
- Configurable CORS
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import numpy as np
import os
import base64
from io import BytesIO
from PIL import Image
import threading
import time
import logging
from functools import wraps

from neural_network import NeuralNetwork, load_mnist_data, is_gpu_enabled, GPU_AVAILABLE

# =============================================================================
# Configuration from Environment Variables
# =============================================================================

DEBUG = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
PORT = int(os.environ.get('PORT', 5000))
HOST = os.environ.get('HOST', '0.0.0.0')
ALLOWED_ORIGINS = os.environ.get('ALLOWED_ORIGINS', '*')  # Comma-separated list or '*'
MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16MB default
DATA_DIR = os.environ.get('DATA_DIR', os.path.dirname(os.path.abspath(__file__)))
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()

# =============================================================================
# Logging Configuration
# =============================================================================

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# Flask Application Setup
# =============================================================================

app = Flask(__name__, static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Configure CORS
if ALLOWED_ORIGINS == '*':
    CORS(app)
    logger.warning("CORS is set to allow all origins. Consider restricting in production.")
else:
    origins = [origin.strip() for origin in ALLOWED_ORIGINS.split(',')]
    CORS(app, origins=origins)
    logger.info(f"CORS configured for origins: {origins}")

# Configure rate limiting
RATE_LIMIT_ENABLED = os.environ.get('RATE_LIMIT_ENABLED', 'true').lower() == 'true'
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
    enabled=RATE_LIMIT_ENABLED,
)


# =============================================================================
# Security Headers
# =============================================================================

@app.after_request
def add_security_headers(response):
    """Add security headers to all responses."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return response

# =============================================================================
# Thread-Safe State Management
# =============================================================================

class TrainingState:
    """Thread-safe training state manager."""

    def __init__(self):
        self._lock = threading.RLock()
        self._is_training = False
        self._progress = []
        self._network = None
        self._results = None

    @property
    def is_training(self):
        with self._lock:
            return self._is_training

    @is_training.setter
    def is_training(self, value):
        with self._lock:
            self._is_training = value

    @property
    def network(self):
        with self._lock:
            return self._network

    @network.setter
    def network(self, value):
        with self._lock:
            self._network = value

    @property
    def results(self):
        with self._lock:
            return self._results

    @results.setter
    def results(self, value):
        with self._lock:
            self._results = value

    def add_progress(self, info):
        with self._lock:
            self._progress.append(info)
            # Keep only last 100 progress entries to prevent memory growth
            if len(self._progress) > 100:
                self._progress = self._progress[-100:]

    def get_progress(self, limit=20):
        with self._lock:
            return self._progress[-limit:] if self._progress else []

    def reset(self):
        with self._lock:
            self._progress = []
            self._results = None


training_state = TrainingState()

# =============================================================================
# Input Validation
# =============================================================================

class ValidationError(Exception):
    """Custom validation error."""
    pass


def validate_int(value, name, min_val=None, max_val=None, default=None):
    """Validate and convert to integer with bounds checking."""
    if value is None:
        if default is not None:
            return default
        raise ValidationError(f"{name} is required")

    try:
        result = int(value)
    except (ValueError, TypeError):
        raise ValidationError(f"{name} must be a valid integer")

    if min_val is not None and result < min_val:
        raise ValidationError(f"{name} must be at least {min_val}")
    if max_val is not None and result > max_val:
        raise ValidationError(f"{name} must be at most {max_val}")

    return result


def validate_float(value, name, min_val=None, max_val=None, default=None):
    """Validate and convert to float with bounds checking."""
    if value is None:
        if default is not None:
            return default
        raise ValidationError(f"{name} is required")

    try:
        result = float(value)
    except (ValueError, TypeError):
        raise ValidationError(f"{name} must be a valid number")

    if min_val is not None and result < min_val:
        raise ValidationError(f"{name} must be at least {min_val}")
    if max_val is not None and result > max_val:
        raise ValidationError(f"{name} must be at most {max_val}")

    return result


def validate_choice(value, name, choices, default=None):
    """Validate value is one of allowed choices."""
    if value is None:
        if default is not None:
            return default
        raise ValidationError(f"{name} is required")

    if value not in choices:
        raise ValidationError(f"{name} must be one of: {', '.join(map(str, choices))}")

    return value


def validate_digit(value, name, default=None):
    """Validate digit is 0-9."""
    return validate_int(value, name, min_val=0, max_val=9, default=default)


def require_json(f):
    """Decorator to require JSON content type."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        return f(*args, **kwargs)
    return decorated


# =============================================================================
# Helper Functions
# =============================================================================

def get_data_path(small_dataset: bool, train: bool) -> str:
    """Get the path to the appropriate dataset."""
    if train:
        filename = 'mnist_train_100.csv' if small_dataset else 'mnist_train.csv'
    else:
        filename = 'mnist_test_10.csv' if small_dataset else 'mnist_test.csv'

    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {filename}")
    return path


# =============================================================================
# API Routes
# =============================================================================

@app.route('/')
def index():
    """Serve the main page."""
    return send_from_directory('static', 'index.html')


@app.route('/api/health', methods=['GET'])
@limiter.exempt
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'is_training': training_state.is_training,
        'has_model': training_state.network is not None,
        'has_results': training_state.results is not None,
        'gpu_available': GPU_AVAILABLE
    })


@app.route('/api/status', methods=['GET'])
@limiter.exempt
def get_status():
    """Get current training status."""
    return jsonify({
        'is_training': training_state.is_training,
        'progress': training_state.get_progress(20),
        'has_results': training_state.results is not None
    })


@app.route('/api/train', methods=['POST'])
@limiter.limit("5 per minute")
@require_json
def train_network():
    """Start training the neural network."""
    if training_state.is_training:
        logger.warning("Training request rejected: training already in progress")
        return jsonify({'error': 'Training already in progress'}), 400

    data = request.json

    try:
        # Validate and parse parameters
        n_epochs = validate_int(data.get('epochs'), 'epochs', min_val=1, max_val=10000, default=100)
        learning_rate = validate_float(data.get('learningRate'), 'learningRate', min_val=0.0001, max_val=10.0, default=0.01)
        use_small_data = bool(data.get('useSmallData', True))
        use_gpu = bool(data.get('useGpu', True))  # Use GPU by default if available
        backprop_method = validate_choice(data.get('backpropMethod'), 'backpropMethod', ['cb', 'uhb'], default='cb')
        hidden_u = validate_int(data.get('hiddenU'), 'hiddenU', min_val=1, max_val=1024, default=128)
        hidden_v = validate_int(data.get('hiddenV'), 'hiddenV', min_val=1, max_val=1024, default=64)
        hidden_w = validate_int(data.get('hiddenW'), 'hiddenW', min_val=1, max_val=1024, default=32)
        digit_a = validate_digit(data.get('digitA'), 'digitA', default=0)
        digit_b = validate_digit(data.get('digitB'), 'digitB', default=1)
        digit_c = validate_digit(data.get('digitC'), 'digitC', default=2)
        digit_d = validate_digit(data.get('digitD'), 'digitD', default=3)

        # Validate unique digits
        digits = [digit_a, digit_b, digit_c, digit_d]
        if len(set(digits)) != 4:
            raise ValidationError("All four digits must be unique")

    except ValidationError as e:
        logger.warning(f"Training request validation failed: {e}")
        return jsonify({'error': str(e)}), 400

    # Reset state
    training_state.reset()
    training_state.is_training = True

    gpu_status = "GPU" if (use_gpu and GPU_AVAILABLE) else "CPU"
    logger.info(f"Starting training [{gpu_status}]: epochs={n_epochs}, lr={learning_rate}, method={backprop_method}, "
                f"hidden=({hidden_u},{hidden_v},{hidden_w}), digits={digits}")

    def progress_callback(info):
        progress_data = {
            'epoch': info['epoch'],
            'totalEpochs': info['total_epochs'],
            'loss': float(info['loss']),
            'accuracy': float(info['accuracy']),
            'timestamp': time.time()
        }
        # Include intra-epoch sample progress if available
        if 'samples_done' in info:
            progress_data['samples_done'] = info['samples_done']
            progress_data['total_samples'] = info['total_samples']
        training_state.add_progress(progress_data)

    def train_thread():
        try:
            # Create network
            nn = NeuralNetwork(
                hidden_layers=(hidden_u, hidden_v, hidden_w),
                learning_rate=learning_rate,
                digits_to_classify=(digit_a, digit_b, digit_c, digit_d),
                use_gpu=use_gpu
            )
            training_state.network = nn

            # Load training data
            train_path = get_data_path(use_small_data, train=True)
            X_train, labels_train = load_mnist_data(train_path)
            Y_train = nn.one_hot_encode(labels_train)

            logger.info(f"Loaded training data: {X_train.shape[1]} samples")

            # Train
            history = nn.train(
                X_train, Y_train,
                n_epochs=n_epochs,
                backprop_method=backprop_method,
                progress_callback=progress_callback
            )

            # Evaluate on training data
            train_results = nn.evaluate(X_train, Y_train)

            # Load and evaluate test data
            test_path = get_data_path(use_small_data, train=False)
            X_test, labels_test = load_mnist_data(test_path)
            Y_test = nn.one_hot_encode(labels_test)
            test_results = nn.evaluate(X_test, Y_test)

            logger.info(f"Training complete: train_acc={train_results['accuracy']:.4f}, "
                       f"test_acc={test_results['accuracy']:.4f}")

            # Store results
            training_state.results = {
                'training': train_results,
                'testing': test_results,
                'history': history,
                'config': {
                    'epochs': n_epochs,
                    'learningRate': learning_rate,
                    'backpropMethod': backprop_method,
                    'hiddenLayers': [hidden_u, hidden_v, hidden_w],
                    'digits': digits,
                    'useSmallData': use_small_data
                }
            }

        except FileNotFoundError as e:
            error_msg = f"Dataset error: {e}"
            logger.error(error_msg)
            training_state.add_progress({
                'error': error_msg,
                'timestamp': time.time()
            })
        except Exception as e:
            error_msg = f"Training failed: {e}"
            logger.exception(error_msg)
            training_state.add_progress({
                'error': str(e),
                'timestamp': time.time()
            })
        finally:
            training_state.is_training = False

    # Start training in background thread
    thread = threading.Thread(target=train_thread, daemon=True)
    thread.start()

    return jsonify({'message': 'Training started'})


@app.route('/api/results', methods=['GET'])
@limiter.exempt
def get_results():
    """Get training and testing results."""
    results = training_state.results
    if results is None:
        return jsonify({'error': 'No results available'}), 404

    return jsonify(results)


@app.route('/api/predict', methods=['POST'])
@limiter.limit("30 per minute")
@require_json
def predict():
    """Make prediction on a drawn digit."""
    nn = training_state.network
    if nn is None:
        return jsonify({'error': 'No trained model available'}), 400

    data = request.json
    image_data = data.get('image')

    if not image_data:
        return jsonify({'error': 'No image provided'}), 400

    if not isinstance(image_data, str):
        return jsonify({'error': 'Image must be a base64 string'}), 400

    # Limit base64 string size (roughly 1MB decoded)
    if len(image_data) > 1.5 * 1024 * 1024:
        return jsonify({'error': 'Image too large'}), 400

    try:
        # Decode base64 image
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]

        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))

        # Convert to grayscale and resize to 28x28
        image = image.convert('L')
        image = image.resize((28, 28), Image.Resampling.LANCZOS)

        # Invert (white digit on black background to black digit on white)
        image_array = 255 - np.array(image)

        # Normalize and flatten
        X = image_array.flatten() / 255.0

        # Predict
        pred_class, probabilities = nn.predict(X)

        # Get digit label
        predicted_digit = nn.get_digit_label(pred_class[0])

        logger.debug(f"Prediction made: {predicted_digit}")

        return jsonify({
            'prediction': predicted_digit,
            'class_index': int(pred_class[0]),
            'probabilities': {
                str(nn.A): float(probabilities[0, 0]),
                str(nn.B): float(probabilities[1, 0]),
                str(nn.C): float(probabilities[2, 0]),
                str(nn.D): float(probabilities[3, 0]),
                'None': float(probabilities[4, 0])
            },
            'processed_image': image_array.tolist()
        })
    except base64.binascii.Error:
        logger.warning("Invalid base64 image data received")
        return jsonify({'error': 'Invalid base64 image data'}), 400
    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({'error': 'Prediction failed'}), 500


@app.route('/api/sample_images', methods=['GET'])
def get_sample_images():
    """Get sample images from the test set."""
    nn = training_state.network
    if nn is None:
        return jsonify({'error': 'No trained model available'}), 400

    try:
        test_path = get_data_path(True, train=False)
        X_test, labels_test = load_mnist_data(test_path)

        # Get up to 10 samples
        n_samples = min(10, X_test.shape[1])
        samples = []

        for i in range(n_samples):
            image_data = (X_test[:, i] * 255).astype(int).tolist()
            pred_class, probs = nn.predict(X_test[:, i:i+1])

            samples.append({
                'image': image_data,
                'true_label': int(labels_test[i]),
                'predicted': nn.get_digit_label(pred_class[0]),
                'probabilities': probs[:, 0].tolist()
            })

        return jsonify({'samples': samples})
    except FileNotFoundError as e:
        logger.error(f"Sample images failed: {e}")
        return jsonify({'error': 'Test dataset not found'}), 500
    except Exception as e:
        logger.exception("Failed to get sample images")
        return jsonify({'error': 'Failed to load sample images'}), 500


# =============================================================================
# Error Handlers
# =============================================================================

@app.errorhandler(400)
def bad_request(e):
    return jsonify({'error': 'Bad request'}), 400


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(413)
def request_too_large(e):
    return jsonify({'error': 'Request too large'}), 413


@app.errorhandler(429)
def rate_limit_exceeded(e):
    logger.warning(f"Rate limit exceeded: {request.remote_addr}")
    return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429


@app.errorhandler(500)
def internal_error(e):
    logger.exception("Internal server error")
    return jsonify({'error': 'Internal server error'}), 500


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    logger.info(f"Starting MNIST Neural Network server on {HOST}:{PORT}")
    logger.info(f"Debug mode: {DEBUG}")
    logger.info(f"Data directory: {DATA_DIR}")

    if DEBUG:
        logger.warning("Running in DEBUG mode - do not use in production!")

    app.run(debug=DEBUG, host=HOST, port=PORT, threaded=True)
