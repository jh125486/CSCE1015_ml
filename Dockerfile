# Use Python slim for smaller footprint
FROM python:3.12-slim

# Install dependencies
RUN pip install torch torchvision Pillow flask

# Copy the application files
WORKDIR /app
COPY classify.py /app/classify.py

# Run the Flask app
CMD ["python", "classify.py"]

