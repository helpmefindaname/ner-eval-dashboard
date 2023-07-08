FROM python:3.10.1
ARG BUILD=prod
ARG BUILD_VERSION=0.1.0
ENV POETRY_VERSION=1.3.1

RUN apt-get update && apt-get install -y g++ cmake

ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    POETRY_HOME="/opt/poetry"
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

WORKDIR /app
ENTRYPOINT ["ner_eval_dashboard"]

RUN curl -sSL https://install.python-poetry.org | python -

COPY . .

RUN poetry install --without dev
RUN poetry version $BUILD_VERSION

RUN if [ $BUILD = "test" ] ; then poetry install; fi

RUN python -c 'import nltk;nltk.download("punkt")'

