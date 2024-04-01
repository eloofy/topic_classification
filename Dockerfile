FROM python:3.10

RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive \
  && apt-get install -y net-tools netcat-traditional curl \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN mkdir --parents /opt/app
COPY pyproject.toml /opt/app/pyproject.toml
COPY poetry.lock /opt/app/poetry.lock
COPY poetry.toml /opt/app/poetry.toml

WORKDIR /opt/app

RUN pip install poetry \
    && poetry install --only main --no-root

COPY . /opt/app/
COPY entrypoint.sh /opt/app/entrypoint.sh

RUN chmod +x /opt/app/entrypoint.sh
