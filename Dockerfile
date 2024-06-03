# Use the official TensorFlow image as a parent image
FROM tensorflow/tensorflow:latest

# Set the working directory in the container
WORKDIR /app

# Copy the source code to the container
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

# Expose the port
EXPOSE 4000

# Run the application
CMD ["/bin/bash"]
