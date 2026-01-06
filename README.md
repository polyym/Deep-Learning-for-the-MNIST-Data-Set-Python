# MNIST Neural Network Web Application

A production-ready Python web application for handwritten digit recognition, converted from the original MATLAB implementation for the MA2647 Artificial Neural Network project.

## Live Demo

**[Try it online](https://mnist-neural-network.onrender.com)** (hosted on Render's free tier - CPU only)

> **Note:** The hosted version runs on CPU only. For GPU-accelerated training, run locally with an NVIDIA GPU (see instructions below).

## Features

- **Scientific Paper UI**: Clean, academic-style interface with equations and methodology
- **GPU Acceleration**: Optional CUDA support via CuPy for 10-50x faster training
- **Configurable Architecture**: 3 hidden layers with user-defined neuron counts
- **Two Backpropagation Methods**:
  - Calculus-Based (CB) - uses chain rule for gradient computation
  - Unscaled Heuristic (UHB) - simplified error backpropagation
- **Real-time Training Progress**: Watch loss curves and accuracy metrics update live
- **Confusion Matrices**: Visualize classification performance
- **Draw & Predict**: Test the trained model by drawing digits
- **Production Ready**: Thread-safe, rate limiting, security headers, input validation

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- (Optional) NVIDIA GPU with CUDA for accelerated training

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/polyym/Deep-Learning-for-the-MNIST-Data-Set-Python.git
cd Deep-Learning-for-the-MNIST-Data-Set-Python

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
python app.py

# 5. Open your browser to:
#    http://localhost:5000
```

### Quick Test
1. Leave all settings at defaults
2. Click **"Begin Training"**
3. Wait for training to complete (~10 seconds with small dataset)
4. Navigate to **Section 4: Results** to view loss curves and confusion matrices
5. Try **Section 5: Interactive Demo** to test your own handwriting

---

## GPU Acceleration (Local Setup)

The hosted version on Render runs on CPU only. For significantly faster training, run the application locally with GPU acceleration.

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
   # For CUDA 12.x
   pip install cupy-cuda12x

   # For CUDA 11.x
   pip install cupy-cuda11x
   ```

3. **Verify GPU is detected:**
   ```bash
   python -c "from neural_network import GPU_AVAILABLE; print(f'GPU Available: {GPU_AVAILABLE}')"
   ```

4. **Start the server:**
   ```bash
   python app.py
   ```

### Performance Comparison

| Dataset | CPU (i7-9700K) | GPU (RTX 2080 Ti) | Speedup |
|---------|----------------|-------------------|---------|
| Small (100 samples) | ~0.5s/epoch | ~0.1s/epoch | 5x |
| Full (60K samples) | ~35s/epoch | ~3s/epoch | 12x |

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
- Python 3.11 runtime
- Gunicorn production server with 2 workers and 4 threads
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
| `DATA_DIR` | Path to CSV data files | Application directory |
| `LOG_LEVEL` | Logging level | `INFO` |
| `RATE_LIMIT_ENABLED` | Enable API rate limiting | `true` |

### Local Production Server

```bash
# Using Gunicorn (Linux/macOS)
gunicorn app:app --bind 0.0.0.0:5000 --workers 2 --threads 4

# Using Waitress (Windows)
pip install waitress
waitress-serve --port=5000 app:app
```

---

## Project Structure

```
Deep-Learning-for-the-MNIST-Data-Set-Python/
├── app.py                 # Flask backend server
├── neural_network.py      # Neural network implementation (NumPy/CuPy)
├── requirements.txt       # Python dependencies
├── render.yaml            # Render deployment configuration
├── .env.example           # Environment variable template
├── test_app.py            # Unit tests (40 tests)
├── static/
│   └── index.html         # React frontend (single-file)
├── mnist_train_100.csv    # Small training set (100 samples)
├── mnist_test_10.csv      # Small test set (10 samples)
├── mnist_train.csv        # Full training set (60,000 samples)
├── mnist_test.csv         # Full test set (10,000 samples)
└── README.md
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
| `/api/status` | GET | - | Get training status and progress |
| `/api/results` | GET | - | Get training/testing results |
| `/api/predict` | POST | 30/min | Predict a drawn digit |
| `/api/sample_images` | GET | - | Get sample test images |

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

## Results from Original MATLAB Implementation

| Method | Epochs | Learning Rate | Train Acc. | Test Acc. |
|--------|--------|---------------|------------|-----------|
| UHB | 200 | 0.001 | 100% | 97.96% |
| CB | 100 | 0.01 | 100% | 98.11% |

## Testing

Run the test suite:

```bash
pytest test_app.py -v
```

With coverage:

```bash
pytest test_app.py --cov=app --cov-report=term-missing
```

## Troubleshooting

### Port already in use
Set a different port using the `PORT` environment variable:
```bash
PORT=5001 python app.py
```

### Missing dependencies
```bash
pip install flask flask-cors flask-limiter numpy pillow gunicorn
```

### Training is slow
- Use the **"Small"** dataset option for quick testing
- Enable GPU acceleration if you have an NVIDIA GPU (see GPU section above)
- The full dataset (60K samples) takes ~30-40s/epoch on CPU vs ~3s/epoch on GPU
- Training 50 epochs on the full dataset takes ~30 minutes on CPU

### Drawing prediction not working
- Ensure you've trained a model first
- Draw digits clearly in the center of the canvas
- Use thick strokes

### Getting the Full Training Dataset
The included `mnist_train_100.csv` has only 100 samples. For the full 60,000 training samples:
1. Download from [Kaggle MNIST in CSV](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
2. Save as `mnist_train.csv` in the project folder
3. Select "Full (60K)" in the Dataset option

## Links

- **Python Version**: [GitHub Repository](https://github.com/polyym/Deep-Learning-for-the-MNIST-Data-Set-Python)
- **Original MATLAB**: [GitHub Repository](https://github.com/polyym/Deep-Learning-for-the-MNIST-Data-Set)
- **Project Report**: [PDF](https://github.com/polyym/Deep-Learning-for-the-MNIST-Data-Set/blob/main/1923114_Report.pdf)
- **Video Walkthrough**: [YouTube](https://www.youtube.com/watch?v=AcSmXXuit6k)

## Credits

Converted from MATLAB implementation by Hadi K for the MA2647 Deep Learning course.

## License

Educational use only.
