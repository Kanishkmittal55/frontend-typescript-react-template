# Makefile

# Variables
COMPOSE_FILE = docker-compose.yml

NEO4J_USER=neo4j
NEO4J_PASSWORD=your12345# Replace with your actual password

# Targets
up:
	docker-compose -f $(COMPOSE_FILE) down
	docker-compose -f $(COMPOSE_FILE) up -d

down:
	docker-compose -f $(COMPOSE_FILE) down

up-reset:
	@echo "Stopping and removing containers and associated images..."
	docker-compose -f $(COMPOSE_FILE) down --rmi local --volumes --remove-orphans
	@echo "Starting fresh containers..."
	docker-compose -f $(COMPOSE_FILE) up -d

clean:
	@echo "Removing all Docker images, containers, and volumes..."
	docker system prune -a --volumes -f
	@echo "Cleanup complete."

insert:
	docker exec -i $(CONTAINER_ID) bin/cypher-shell -u $(NEO4J_USER) -p $(NEO4J_PASSWORD) < $(FILE)

help:
	@echo "Makefile targets:"
	@echo "  make up       - Start the containers (docker-compose up)"
	@echo "  make down     - Stop and remove the containers (docker-compose down)"
	@echo "  make up-reset - Reset by removing containers and images related to this compose file, then start containers"
	@echo "  make clean    - Remove all Docker images, containers, and volumes"
	@echo "  make help     - Show this help message"
	@echo "  make insert     - insert a cypher file into the neo4j db - usage - make insert file=<filename>"
