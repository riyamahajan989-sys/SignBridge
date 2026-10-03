FROM python:3.12-slim

# Install Linux libraries required by MediaPipe/OpenCV
RUN apt-get update && apt-get install -y \
    libgles2 \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 app:app"]