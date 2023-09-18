FROM python:3.10
    
ENV DASH_DEBUG_MODE=False
ENV PYTHONUNBUFFERED 1

RUN apt-get update  && apt-get install -y  g++ llvm lcov 
RUN apt-get install --yes gdal-bin
RUN apt-get install --yes libgdal-dev

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

RUN mkdir -p /usr/src/app/
WORKDIR /usr/src/app/

COPY lib/ /usr/src/app/lib
COPY output_app/ /usr/src/app/output_app
COPY data/taxi_zones/ /usr/src/app/data/taxi_zones

COPY output_app/requirements.txt /usr/src/app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080
 
WORKDIR ./output_app
CMD ["python", "app.py"]