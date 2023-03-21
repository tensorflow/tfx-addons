ARG PLATFORM=cpu

FROM gcr.io/tfx-oss-public/tfx:latest

WORKDIR /tfx-addons
RUN mkdir -p /tfx-addons/tfx_addons
ADD __init__.py /tfx-addons/tfx_addons
COPY ./ ./tfx_addons/predictions_to_bigquery

ENV PYTHONPATH="/tfx-addons:${PYTHONPATH}"
