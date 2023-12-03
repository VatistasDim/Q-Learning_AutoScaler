FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY / .

EXPOSE 8501
EXPOSE 8082

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8501"]
