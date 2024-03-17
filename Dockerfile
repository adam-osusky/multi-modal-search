# Use nvidia container for gpu support
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

# Install Python 3 and pip
RUN apt-get update \
    && apt-get install -y \
        python3 \
        python3-pip

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install sys dependencies. Tesseract OCR, opencv depends, poppler for unstructured,
# set TESSDATA_PREFIX environment variable, pciutils and lshw for ollama gpu
RUN apt-get update \
    && apt-get install -y \
        tesseract-ocr \
        ffmpeg \
        libsm6 \
        libxext6 \
        poppler-utils \
        curl \
        pciutils \
        lshw \
    && export TESSDATA_PREFIX=/usr/share/tessdata

# Copy the mulmod project
COPY . /app/

# Install Poetry and dependencies
RUN pip install poetry \
    && poetry config virtualenvs.create false \
    && poetry install --only main

# Install ollama and pull models
RUN curl -fsSL https://ollama.com/install.sh | sh