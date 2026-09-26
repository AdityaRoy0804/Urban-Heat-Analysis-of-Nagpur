FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/
COPY data/ ./data/

EXPOSE 8000

CMD ["uvicorn", "nagpur_uhi.api.main:app", "--host", "0.0.0.0", "--port", "8000"]