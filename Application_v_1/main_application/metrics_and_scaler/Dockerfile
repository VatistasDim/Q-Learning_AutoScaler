FROM python:3.9

WORKDIR /app

COPY  / .

RUN pip install requests docker numpy

VOLUME /var/run/docker.sock

CMD [ "python", "/app/docker_service_autoscaler.py" ]
