FROM python:3.10.1
ARG BUILD=prod
ENV PIP_VERSION=22.1.2
# should be fixed but as high as possible

RUN apt-get update && apt-get install -y g++ cmake

RUN pip install pip==$PIP_VERSION

WORKDIR /app
ENTRYPOINT ["ner_eval_dashboard"]

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN pip install -e . --no-deps

RUN if [ $BUILD = "test" ] ; then pip install -r requirements-dev.txt; fi



