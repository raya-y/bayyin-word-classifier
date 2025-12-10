FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY config.yaml .

# Expose port (HF Spaces uses 7860 by default)
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application (HF Spaces expects port 7860)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]

