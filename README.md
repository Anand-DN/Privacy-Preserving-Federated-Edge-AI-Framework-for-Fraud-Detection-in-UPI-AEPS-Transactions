# Federated Fraud Detection System

Privacy-Preserving UPI Fraud Detection with Differential Privacy. Access the Live ink here:  https://federated-learning-upi-fraud-detection.streamlit.app/

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the prediction pipeline
python main.py

# Launch Streamlit dashboard
streamlit run app.py
```

## Deployment

### Streamlit Cloud (Recommended)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Set:
   - Main file: `app.py`
   - Python version: 3.12

### Local

```bash
streamlit run app.py --server.port 8501
```

### Docker

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

```bash
docker build -t fraud-detection .
docker run -p 8501:8501 fraud-detection
```

## Features

- Federated Learning with Differential Privacy
- Secure Aggregation with Byzantine tolerance
- Multi-model comparison (LR/RF/MLP)
- Per-bank evaluation
- Privacy budget tracking
- Real-time fraud prediction

## Architecture

- **Banks (Clients)**: Train models locally on private data
- **Central Server**: Aggregates model weights (not data)
- **Differential Privacy**: Gaussian noise prevents data reconstruction
- **Privacy Budget**: Tracks cumulative epsilon spent
