.PHONY: build up down logs

build:
	./gradlew clean test pluginZip

up:
	docker compose up --build --detach --wait

down:
	docker compose down --volumes --remove-orphans

logs:
	docker compose logs --follow elasticsearch