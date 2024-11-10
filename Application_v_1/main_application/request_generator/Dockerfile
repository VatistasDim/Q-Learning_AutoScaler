FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY poisson_request_generator.py .

CMD ["python", "poisson_request_generator.py"]
