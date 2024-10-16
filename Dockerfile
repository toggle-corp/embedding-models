FROM python:3.12-slim-bullseye

LABEL maintainer="TC Developers"

ENV PYTHONUNBUFFERED 1

WORKDIR /code

RUN apt-get update -y && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock /code/

# Upgrade pip and install python packages for code
RUN pip install --upgrade --no-cache-dir pip poetry \
    && poetry --version \
    # Configure to use system instead of virtualenvs
    && poetry config virtualenvs.create false \
    && poetry install --no-root \
    # Remove installer
    && rm -rf /root/.cache/pypoetry \
    && pip uninstall -y poetry virtualenv-clone virtualenv

COPY . /code/