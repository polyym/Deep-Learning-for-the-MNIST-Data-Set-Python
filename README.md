# MNIST Neural Network Web Application

A production-ready Python web application for handwritten digit recognition, ported from my own MATLAB implementation for the MA2647 Deep Learning course.

## Live Demo

**[Try it online](https://mnist-neural-network.onrender.com)** (hosted on Render's free tier - CPU only)

> **Note:** The hosted version runs on CPU only. For GPU-accelerated training, run locally with an NVIDIA GPU (see instructions below).

## Features

- **Scientific Paper UI**: Clean, academic-style interface with equations and methodology
- **Configurable Architecture**: 3 hidden layers with user-defined neuron counts
- **Two Backpropagation Methods**:
  - Calculus-Based (CB) - uses chain rule for gradient computation
  - Unscaled Heuristic (UHB) - simplified error backpropagation
- **Real-time Training Progress**: Watch loss and accuracy update live; cancel a run mid-training with one click
- **Optional GPU Backend**: CuPy/CUDA path for larger networks (see the GPU section; CPU is typically faster at the default hidden-layer sizes)
- **Confusion Matrices**: Visualise classification performance
- **Draw & Predict**: Test the trained model by drawing digits. Inputs are preprocessed to match the MNIST convention (bounding-box crop → 20×20 scale → centre-of-mass centring on a 28×28 canvas)
- **Production Ready**: Thread-safe, rate limiting, security headers, input validation

## Quick Start

### Prerequisites
- Python 3.13 or higher
- [uv](https://docs.astral.sh/uv/) (recommended; it manages the venv and
  installs dependencies automatically, pinned against `uv.lock` for
  reproducibility). One-line install:
  ```bash
  # Linux / macOS
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Windows (PowerShell)
  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- (Optional) NVIDIA GPU with CUDA for accelerated training

### Installation (uv, recommended)

```bash
# 1. Clone the repository
git clone https://github.com/polyym/Deep-Learning-for-the-MNIST-Data-Set-Python.git
cd Deep-Learning-for-the-MNIST-Data-Set-Python

# 2. Start the server. `uv run` creates `.venv/` from pyproject.toml + uv.lock
#    on first run, installs the package, and executes the console script.
uv run mnist-ann

# 3. Open your browser to:
#    http://localhost:5000
```

Subsequent runs reuse the same `.venv/`; `uv` re-checks the lockfile in a
few ms and refuses to drift.

> **GPU users:** the default `uv run` installs the CPU-only base dependencies.
> If you have an NVIDIA GPU with CUDA, also run `uv sync --extra gpu-cuda12`
> (CUDA 12.x) or `uv sync --extra gpu-cuda11` (CUDA 11.x) once to pull CuPy
> into the same `.venv/`. Having CuPy in your system Python is **not enough**,
> since `uv`'s venv is isolated. See [GPU Acceleration](#gpu-acceleration-optional)
> below for full setup.

### Installation (pip fallback)

If you prefer not to install `uv`, the standard pip/venv flow still works:

```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -e .
mnist-ann                   # or: python -m mnist_ann
```

### Quick Test
1. Leave all settings at defaults
2. Click **"Begin Training"**
3. Wait for training to complete (~10 seconds with small dataset)
4. Navigate to **Section 4: Results** to view loss curves and confusion matrices
5. Try **Section 5: Interactive Demo** to test your own handwriting

---

## GPU Acceleration (Optional)

This implementation uses **online SGD**, one forward/backward pass and weight update per sample. With the default hidden-layer sizes (64/32/16), per-operation GPU kernel-launch overhead dominates the actual compute, so **CPU is typically faster than GPU in the default configuration**. GPU becomes worthwhile once you enlarge hidden layers significantly (roughly 512+ neurons per layer).

The hosted version on Render runs on CPU only; running locally with or without GPU is a question of whether you want the CuPy backend available, not a requirement for reasonable performance.

### Requirements

- NVIDIA GPU (GTX 1060 or better recommended)
- CUDA Toolkit 11.x or 12.x
- cuDNN (optional but recommended)

### Setup

1. **Check your CUDA version:**
   ```bash
   nvidia-smi
   ```
   Look for "CUDA Version" in the output (e.g., `CUDA Version: 12.2`).

2. **Install CuPy matching your CUDA version:**
   ```bash
   # uv (recommended)
   uv sync --extra gpu-cuda12   # CUDA 12.x
   uv sync --extra gpu-cuda11   # CUDA 11.x

   # pip fallback
   pip install ".[gpu-cuda12]"  # or ".[gpu-cuda11]"
   ```

3. **Verify GPU is detected:**
   ```bash
   uv run python -c "from mnist_ann import GPU_AVAILABLE; print(f'GPU Available: {GPU_AVAILABLE}')"
   ```

4. **Start the server:**
   ```bash
   uv run mnist-ann
   ```

### Approximate Training Times

Exact numbers depend on hardware; the takeaway is the *relative trend*:

| Scenario | Hidden Layers | CPU | GPU (RTX 2080 Ti) |
|----------|---------------|------|-------------------|
| Small dataset (100), 50 epochs | 64/32/16 | ~1–2 s | ~30–60 s |
| Full dataset (60K), 50 epochs | 64/32/16 | ~15–25 min | hours (kernel-launch bound) |
| Full dataset (60K), 50 epochs | 512/256/128 | much slower | typically faster than CPU |

For the defaults shipped in the UI, **just use CPU**. Switch to GPU only if you've widened the network enough that the per-op compute dominates launch overhead.

### Troubleshooting GPU Issues

**GPU not detected:**
```bash
# Check CUDA installation
nvidia-smi

# Test CuPy directly
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"
```

**CuPy installation fails:**
- Ensure CUDA Toolkit is installed (not just the driver)
- On Windows, install [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- Try pre-built wheels: `pip install cupy-cuda12x --no-cache-dir`

**Out of memory errors:**
- Reduce batch size (this implementation uses online learning, so memory usage is minimal)
- Close other GPU applications

---

## Deployment

### Render (Recommended for Hosting)

This application is configured for deployment on [Render](https://render.com):

1. Push your code to GitHub
2. Create a new Web Service on Render
3. Connect your GitHub repository
4. Render will auto-detect `render.yaml` and configure everything

The `render.yaml` configuration includes:
- Python 3.13 runtime
- `pip install -e .` build (editable install so `config.py` can locate the
  repo's `static/` and `data/` directories at runtime)
- Gunicorn production server using the factory entry point
  `mnist_ann.app:create_app()`, 1 worker, 8 threads (the in-process
  `TrainingState` singleton would be duplicated across workers, so a
  multi-worker deploy would split training/model state per request)
- 120-second timeout for long training requests
- Automatic port configuration

> **Free Tier Limitations:** Render's free tier runs on CPU only and may spin down after inactivity. The first request after spin-down takes ~30 seconds.

### Environment Variables

Configure the application using environment variables (see `.env.example`):

| Variable | Description | Default |
|----------|-------------|---------|
| `FLASK_DEBUG` | Enable debug mode | `false` |
| `PORT` | Server port | `5000` |
| `HOST` | Server host | `0.0.0.0` |
| `ALLOWED_ORIGINS` | CORS origins (comma-separated or `*`) | `*` |
| `MAX_CONTENT_LENGTH` | Max request size in bytes | `16777216` |
| `DATA_DIR` | Path to the MNIST CSV directory | `<repo>/data/` (or `$CWD/data/`) |
| `STATIC_DIR` | Path to the frontend static files | `<repo>/static/` (or `$CWD/static/`) |
| `LOG_LEVEL` | Logging level | `INFO` |
| `RATE_LIMIT_ENABLED` | Enable API rate limiting | `true` |

### Local Production Server

```bash
# Using Gunicorn (Linux/macOS) -- note the factory invocation
uv run gunicorn "mnist_ann.app:create_app()" --bind 0.0.0.0:5000 --workers 1 --threads 8
# or, outside uv:
# gunicorn "mnist_ann.app:create_app()" --bind 0.0.0.0:5000 --workers 1 --threads 8

# Using Waitress (Windows) -- ad-hoc, without adding it as a project dep
uv run --with waitress waitress-serve --port=5000 --call mnist_ann.app:create_app
```

---

## Project Structure

```
Deep-Learning-for-the-MNIST-Data-Set-Python/
├── pyproject.toml              # Package metadata, dependencies, pytest config
├── uv.lock                     # Lockfile (used by `uv sync`); commit to git
├── render.yaml                 # Render deployment configuration
├── .env.example                # Environment variable template
├── .gitignore
├── README.md
│
├── .github/
│   └── workflows/
│       └── tests.yml           # CI: run pytest on push/PR
│
├── src/mnist_ann/              # Application package
│   ├── __init__.py             # Public re-exports + __version__
│   ├── __main__.py             # `python -m mnist_ann` entry
│   ├── app.py                  # Flask app factory (create_app)
│   ├── routes.py               # Blueprint with every /api/* route
│   ├── config.py               # Env config, paths, logging
│   ├── extensions.py           # Flask-Limiter instance
│   ├── state.py                # Thread-safe TrainingState singleton
│   ├── validation.py           # Input validators + ValidationError
│   ├── preprocessing.py        # Canvas drawing -> MNIST-style 28x28 input
│   ├── backend.py              # GPU detection + NumPy/CuPy abstraction
│   ├── progress.py             # Console progress bar
│   ├── data.py                 # CSV loader + dataset path resolution
│   └── network.py              # NeuralNetwork class
│
├── tests/                      # Pytest suite (mirrors the package layout)
│   ├── conftest.py
│   ├── test_app_factory.py
│   ├── test_data.py
│   ├── test_endpoints.py
│   ├── test_errors.py
│   ├── test_network.py
│   ├── test_preprocessing.py
│   ├── test_state.py
│   └── test_validation.py
│
├── data/                       # MNIST CSV datasets
│   ├── mnist_train_100.csv     # Small training set (100 samples)
│   ├── mnist_test_10.csv       # Small test set (10 samples)
│   ├── mnist_train.csv         # Full training set (60,000 samples)
│   └── mnist_test.csv          # Full test set (10,000 samples)
│
└── static/
    └── index.html              # React frontend (single-file, inline Babel)
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| Epochs | Number of training iterations | 50 |
| Learning Rate | Step size for gradient descent | 0.01 |
| Dataset Size | Small (100/10) or Full (60K/10K) | Small |
| Compute Device | GPU (if available) or CPU | CPU |
| Backprop Method | CB (calculus) or UHB (heuristic) | CB |
| Hidden Layer U | Neurons in first hidden layer | 64 |
| Hidden Layer V | Neurons in second hidden layer | 32 |
| Hidden Layer W | Neurons in third hidden layer | 16 |
| Digits A, B, C, D | Which digits to classify (0-9) | 0, 1, 2, 3 |

## Network Architecture

```
Input (784) → H1 (U) → H2 (V) → H3 (W) → Output (5)
              [sigmoid]  [sigmoid]  [sigmoid]  [softmax]
```

- **Input Layer**: 784 neurons (28x28 pixel images)
- **Hidden Layers**: 3 layers with sigmoid activation
- **Output Layer**: 5 neurons with softmax activation
  - Classes 1-4: The four selected digits (A, B, C, D)
  - Class 5: "None" (any other digit)

## API Endpoints

| Endpoint | Method | Rate Limit | Description |
|----------|--------|------------|-------------|
| `/` | GET | - | Serve the web interface |
| `/api/health` | GET | - | Health check with GPU status |
| `/api/train` | POST | 5/min | Start training with configuration |
| `/api/cancel` | POST | 30/min | Request cancellation of an in-flight training run |
| `/api/status` | GET | 600/min | Training status + recent progress tail (polled 2/sec by the UI) |
| `/api/results` | GET | - | Get training/testing results |
| `/api/predict` | POST | 30/min | Predict a drawn digit |
| `/api/sample_images` | GET | 50/hour, 200/day | Get sample test images (inherits default limits) |

## Technical Details

### Forward Propagation

For each layer n:
```
n_n = W_n^T * a_{n-1} + b_n    (pre-activation)
a_n = σ(n_n)                    (activation)
```

### Backpropagation

**Calculus-Based (CB)**:
```
S_n = A_n * W_{n+1} * S_{n+1}
```
Where A_n is the diagonal matrix of activation derivatives.

**Unscaled Heuristic (UHB)**:
```
e_n = W_{n+1} * e_{n+1}
S_n = -2 * A_n * e_n
```

### Weight Update
```
W_n = W_n - lr * a_{n-1} * S_n^T
b_n = b_n - lr * S_n
```

### Loss Function
Cross-entropy loss:
```
L = -Σ y_true * log(y_pred)
```

### Drawing Input Preprocessing

Hand-drawn canvas input is converted to MNIST-style network input before inference:

1. **Greyscale** the canvas PNG at 224×224 and normalise to `[0, 1]`.
2. **Threshold** away the dark-grey background baseline (CSS `#1a1a1a` ≈ 0.10); pixels below 0.15 become `0`.
3. **Bounding-box crop** around the remaining signal pixels.
4. **Scale** the longer side to 20 pixels with LANCZOS resampling, preserving aspect ratio.
5. **Paste** onto a 28×28 canvas.
6. **Shift** so the digit's centre of mass sits at `(13.5, 13.5)`, the same centring convention used when MNIST was originally constructed.

Canvas PNGs already have a bright digit on a dark background (matching MNIST polarity), so **no pixel inversion is applied**. The network sees `bg ≈ 0.0, digit ≈ 1.0`, the same distribution it was trained on.

## Results from Original MATLAB Implementation

| Method | Epochs | Learning Rate | Train Acc. | Test Acc. |
|--------|--------|---------------|------------|-----------|
| UHB | 200 | 0.001 | 100% | 97.96% |
| CB | 100 | 0.01 | 100% | 98.11% |

## Testing

The package ships with a pytest suite covering validators, HTTP endpoints,
state management, error handlers, canvas preprocessing, and the neural
network's math.

```bash
# uv (recommended; auto-installs the [test] extra into .venv on first run)
uv run --extra test pytest
uv run --extra test pytest tests/test_endpoints.py -v
uv run --extra test pytest --cov=mnist_ann --cov-report=term-missing

# pip fallback
pip install -e ".[test]"
pytest
```

`pyproject.toml` sets ``pythonpath = ["src"]`` for pytest, so tests find the
package without a separate editable install, but the editable install is
still needed to get the console script and to `pip install .[test]`.

## Troubleshooting

### Port already in use
Set a different port using the `PORT` environment variable:
```bash
PORT=5001 uv run mnist-ann      # or: PORT=5001 python -m mnist_ann
```

### Missing dependencies
```bash
uv sync                          # or: pip install -e .
```

### Training is slow
- Use the **"Small"** dataset option for quick testing
- On the full dataset (60K samples), roughly **15-25 minutes** for 50 epochs on a modern CPU
- **Prefer CPU for the default layer sizes.** GPU is slower here because of per-op kernel-launch overhead in the online-SGD loop (see the GPU Acceleration section). Enable GPU only if you widen hidden layers significantly
- You can **Cancel Training** mid-run from the UI if you just want to check a checkpoint; results and the trained model are only published once a full run completes, so cancelling leaves the previously trained model intact

### Drawing prediction not working
- Ensure you've trained a model first
- Draw digits clearly in the centre of the canvas
- Use thick strokes

### Getting the Full Training Dataset
The included `data/mnist_train_100.csv` has only 100 samples. For the full 60,000 training samples:
1. Download from [Kaggle MNIST in CSV](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
2. Save as `data/mnist_train.csv`
3. Select "Full (60K)" in the Dataset option

## Links

- **Python Version**: [GitHub Repository](https://github.com/polyym/Deep-Learning-for-the-MNIST-Data-Set-Python)
- **Original MATLAB**: [GitHub Repository](https://github.com/polyym/Deep-Learning-for-the-MNIST-Data-Set)
- **Project Report**: [PDF](https://github.com/polyym/Deep-Learning-for-the-MNIST-Data-Set/blob/main/1923114_Report.pdf)
- **Video Walkthrough**: [YouTube](https://www.youtube.com/watch?v=AcSmXXuit6k)

## AI usage

Claude Sonnet 4.6 Extended (Anthropic) wrote all of the most recent tests and
assisted with some of the documentation, as well as helping me split the files
to better organise this project.
