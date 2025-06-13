FROM python:3.10-slim

# Install sistem dependency yang dibutuhin OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && apt-get clean

# Set working directory
WORKDIR /app

# Copy semua file
COPY . .

# Install dependensi Python
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (optional)
EXPOSE 7860

# Start FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]


