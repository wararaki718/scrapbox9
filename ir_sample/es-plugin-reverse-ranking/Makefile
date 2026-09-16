ES_PORT ?= 9200
ES_URL ?= http://localhost:$(ES_PORT)

.PHONY: build up verify down logs

build:
	./gradlew clean test pluginZip

up:
	docker compose up --build --detach --wait

verify:
	ES_URL=$(ES_URL) ./scripts/verify.sh

down:
	docker compose down --volumes --remove-orphans

logs:
	docker compose logs --follow elasticsearch