# Is Hotdog

A **cutting-edge** containerized app that identifies whether an uploaded image is a hot dog. It uses a small CPU-based PyTorch model and a simple Flask web interface for uploads.

## Features
- Lightweight container with CPU-only PyTorch
- Quick, clear verdict on your uploaded images
- Perfect for demos of containerized machine learning

## Quick Start

1. **Pull the image**:

   docker pull jh125486/ishotdog:latest

2. **Run the container** (mapping port 5050):

   docker run -p 5050:5050 jh125486/ishotdog:latest

3. **Open your browser** at `http://localhost:5050` and execute the advanced AI.

## How It Works
- A pretrained PyTorch model checks if the highest-probability class corresponds to "hot dog."
- The Flask app displays a verdict ("Hotdog!" or "Not hotdog."), plus a preview of the uploaded image.
