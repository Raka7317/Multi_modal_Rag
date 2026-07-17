FROM python:3.11-slim

# ffmpeg -> audio/video processing (moviepy/whisper), poppler -> pdf2image
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg poppler-utils libgl1 build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
