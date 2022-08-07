FROM --platform=linux/amd64 python:3.10-slim
LABEL org.opencontainers.image.authors="rexhaif.io@gmail.com"

WORKDIR /srl

ADD requirements.txt ./requirements.txt
#RUN apt update && apt install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install -r requirements.txt
RUN python -c "from pymystem3 import Mystem; _ = Mystem()" # fetch mystem binary
RUN python -c \
    "import transformers as tr; _ = tr.pipeline('token-classification', model='Rexhaif/rubert-base-srl-seqlabeling')" 
# fetch transformers model

ADD ./app ./app
ADD ./*.toml ./
ADD ./resources ./resources

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 CMD "curl --fail http://localhost:8080/health || exit 1"

CMD uvicorn app.main:app --host=0.0.0.0 --port=8000