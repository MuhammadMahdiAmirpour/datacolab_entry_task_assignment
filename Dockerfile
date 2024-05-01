FROM ubuntu:22.04
LABEL authors="MuhammadMahdiAmirpour"

# Update package lists and install required dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    git \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Set the default Python version to 3.12
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
RUN update-alternatives --config python3

# Create a virtual environment and activate it
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Copy the project files and data files
COPY . /app
COPY ./data/datacolab_dataset /app/data/datacolab_dataset
WORKDIR /app

# Install the required dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set the command to run the Python script
CMD ["python", "src/main.py"]
