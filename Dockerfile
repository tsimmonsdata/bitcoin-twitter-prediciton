FROM python:3.7
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
COPY . /btc_predict
WORKDIR /btc_predict
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["app.py"]