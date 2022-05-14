# syntax=docker/dockerfile:1.2
FROM python:3.9-slim-buster
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
COPY requirements.txt ./
RUN pip install -r requirements.txt
  
