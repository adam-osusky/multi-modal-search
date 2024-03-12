# Use a Python base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy the Poetry files to the container
COPY pyproject.toml poetry.lock README.md /app/

# Copy the rest of the application code to the container
COPY src/ /app/src

# Install Poetry and dependencies
RUN pip install poetry \
    && poetry config virtualenvs.create false \
    && poetry install --only main

# Command to run the application
CMD ["python", "src/mulmod/main.py"]