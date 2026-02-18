# TensorRT-LLM Docker for Jetson
# Ultra-fast LLM inference with TensorRT optimizations
# Based on NVIDIA NGC TensorRT-LLM release image

FROM nvcr.io/nvidia/tensorrt-llm/release:latest

# Install additional utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for model conversion and serving
RUN pip install --no-cache-dir \
    huggingface-hub \
    transformers \
    accelerate \
    flask \
    flask-cors

# Create workspace
WORKDIR /workspace

# Copy helper scripts
COPY run-model.py /workspace/
COPY convert-hf.sh /workspace/

# Create directories for models and engines
RUN mkdir -p /workspace/models /workspace/engines

# Expose API port
EXPOSE 8000

# Default command
CMD ["/bin/bash"]
