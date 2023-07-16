FROM python:3.9

WORKDIR /app

COPY  / .

RUN pip install requests docker

VOLUME /var/run/docker.sock

CMD [ "python", "/app/autoscaler.py" ]
