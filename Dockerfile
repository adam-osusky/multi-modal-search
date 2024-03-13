# Use a Python base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install Tesseract OCR, opencv depends, poppler, and set TESSDATA_PREFIX environment variable
RUN apt-get update \
    && apt-get install -y \
        tesseract-ocr \
        ffmpeg \
        libsm6 \
        libxext6 \
        poppler-utils \
    && export TESSDATA_PREFIX=/usr/share/tessdata

# Copy the Poetry files to the container
COPY pyproject.toml poetry.lock README.md /app/

# Copy folder with resources
COPY resources/ /app/resources/

# Copy the rest of the application code to the container
COPY src/ /app/src

# Install Poetry and dependencies
RUN pip install poetry \
    && poetry config virtualenvs.create false \
    && poetry install --only main

# Command to run the application
CMD ["python", "src/mulmod/main.py"]