FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mnist.py .

# Install Prometheus client library
RUN pip install prometheus_client
RUN pip install --no-cache-dir protobuf==3.20.0

CMD ["python", "mnist.py"]
