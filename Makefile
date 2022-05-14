run:
	docker-compose up --build -d

build:
	docker-compose build

logs:
	docker-compose logs -f --tail=100 -t 

shell:
	docker-compose exec notebook bash
