# Use the specified Python runtime as a parent image
FROM docker.io/nvidia/cuda:12.1.0-cudnn8-devel-ubi8@sha256:f045009cab64c9fda6113b4473ac1c57dfcca65e18ce981bce63f3cddf7b807a

# Set the working directory in the container
WORKDIR /usr/src/app

# Install required packages
RUN apt-get update && apt-get install -y \
    gcc-11 \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variable to use gcc-11
ENV CC=/usr/bin/gcc-11

# Copy the current directory contents into the container
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Set the working directory for the GroundingDINO ops
WORKDIR /usr/src/app/models/GroundingDINO/ops

# Run the setup script and the test script
RUN python setup.py build install
RUN python test.py # This should result in 6 lines of * True

# Install Gradio
RUN pip install gradio

# Change back to the original working directory
WORKDIR /usr/src/app

# Expose the port Gradio will run on
EXPOSE 7860

# Default command to run the Gradio app
CMD ["python", "app.py"]
