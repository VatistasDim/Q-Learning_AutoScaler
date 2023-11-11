FROM python:3.9

WORKDIR /app

COPY  / .

RUN pip install requests docker numpy matplotlib

VOLUME /var/run/docker.sock

CMD [ "python", "-u", "/app/docker_service_autoscaler.py" ]
