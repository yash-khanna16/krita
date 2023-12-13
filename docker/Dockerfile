FROM python:3.10

ADD . /
RUN apt-get update
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install gunicorn
EXPOSE 5000

CMD gunicorn --bind 0.0.0.0:5000 -w 4 --preload main:app --timeout 120
