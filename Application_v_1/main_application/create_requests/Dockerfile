FROM python:3.9

WORKDIR /app

COPY  / .
RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python", "-u", "/app/create_requests.py" ]