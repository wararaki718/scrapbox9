ARG ELASTICSEARCH_VERSION=8.19.21

FROM gradle:8.10.2-jdk21 AS builder

WORKDIR /workspace
COPY --chown=gradle:gradle . .
RUN ./gradlew --no-daemon clean test pluginZip

FROM docker.elastic.co/elasticsearch/elasticsearch:${ELASTICSEARCH_VERSION}

COPY --from=builder --chown=elasticsearch:root \
    /workspace/build/distributions/custom-score-1.0.0.zip \
    /tmp/custom-score.zip
RUN bin/elasticsearch-plugin install --batch file:///tmp/custom-score.zip \
    && rm /tmp/custom-score.zip