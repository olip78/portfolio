FROM python:3.10

ENV PYTHONUNBUFFERED 1

RUN mkdir -p /usr/src/app/
WORKDIR /usr/src/app/

COPY lib/ /usr/src/app/lib
COPY prediction/ /usr/src/app/prediction

COPY requirements.txt /usr/src/app/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080
 
WORKDIR ./prediction
CMD ["python", "ny_taxi_predict.py"] 
