FROM python:3.10

ENV PYTHONUNBUFFERED 1

RUN mkdir -p /usr/src/app/
WORKDIR /usr/src/app/

COPY lib/ /usr/src/app/lib
COPY training/ /usr/src/app/training

COPY requirements.txt /usr/src/app/

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR ./training
CMD ["python", "ny_taxi_training.py"] 
