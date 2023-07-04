FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY cpu_metrics.py .

CMD ["python", "cpu_metrics.py"]
