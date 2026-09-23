FROM python:3.12-slim

WORKDIR /app

# Instalacja bibliotek systemowych wymaganych przez bibliotekę cv2
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN dos2unix requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py snn_model.py franciszek_scnn_53.pth ./
RUN dos2unix main.py snn_model.py

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
