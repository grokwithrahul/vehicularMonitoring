#!/bin/bash

# Build the Docker image
docker build -t vehicle-tracker .

# Run the container with a video file
docker run -v $(pwd)/videos:/app/videos:ro \
           -v $(pwd)/output:/app/output \
           vehicle-tracker \
           python src/object_tracker.py /app/videos/$1
