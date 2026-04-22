FROM python:3.11-slim

# Prevents Python from buffering stdout/stderr — essential for Railway logs
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy bot
COPY bot.py .

# Railway injects PORT — the health server binds to it
EXPOSE 8080

CMD ["python", "bot.py"]
