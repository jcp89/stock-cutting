# syntax=docker/dockerfile:1.2
FROM python:3.9-slim-buster

USER root

RUN mkdir -p /usr/src/app

WORKDIR /usr/src/app

RUN --mount=type=cache,target=/root/buildkit/.cache\
  apt-get update -y && apt-get install -y g++

COPY requirements.txt ./

RUN --mount=type=cache,target=/root/buildkit/.cache\
  pip install -r requirements.txt
