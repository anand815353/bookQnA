.PHONY: test up down

test:
	pytest -q

up:
	docker compose up --build

down:
	docker compose down
