FROM nvcr.io/nvidia/tritonserver:24.06-py3

RUN apt-get update && apt-get install -y git ffmpeg && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python deps
RUN pip3 install --no-cache-dir -r requirements.txt \
 && pip3 install --no-cache-dir git+https://github.com/neonbjb/tortoise-tts.git --no-deps
