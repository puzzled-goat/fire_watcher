-include .env

export IMG_FOLDER_PATH

# Virtual machine (LXC)
export VM_USER VM_HOST VM_PATH

APP_NAME := firewatcher
IMAGE := $(APP_NAME):local
ARCHIVE := $(APP_NAME).tar.gz

.DEFAULT_GOAL := help

help:
	@echo ""
	@echo "Firewatcher commands:"
	@echo "  make build       Build Docker image locally"
	@echo "  make deploy      Build, transfer, load, and start container on LXC"
	@echo "  make logs        Tail container logs on LXC"
	@echo "  make clean       Remove local image and archive"
	@echo ""

# Build local image
build:
	@echo "==> Build image..."
	docker build -t $(IMAGE) .

# Save image to tarball
save: build
	@echo "==> Saving image..."
	docker save $(IMAGE) | gzip > $(ARCHIVE)

# One-command deploy + run on LXC
deploy: save
	@echo "==> Transferring Firewatcher to $(VM_HOST) ..."
	# create folders on host
	ssh $(VM_USER)@$(VM_HOST) "mkdir -p $(VM_PATH)/tmp $(VM_PATH)/models $(VM_PATH)/static"

	# copy local models and static to host
	scp -r models/* $(VM_USER)@$(VM_HOST):$(VM_PATH)/models/
	scp -r static/* $(VM_USER)@$(VM_HOST):$(VM_PATH)/static/
	scp .env $(VM_USER)@$(VM_HOST):$(VM_PATH)/

	# set permissions on host
	ssh $(VM_USER)@$(VM_HOST) "chown -R 1000:1000 $(VM_PATH)/tmp $(VM_PATH)/models $(VM_PATH)/static $(IMG_FOLDER_PATH)"

	scp $(ARCHIVE) $(VM_USER)@$(VM_HOST):$(VM_PATH)/
	scp docker-compose.yml $(VM_USER)@$(VM_HOST):$(VM_PATH)/
	ssh $(VM_USER)@$(VM_HOST) "\
		cd $(VM_PATH) && \
		echo '==> Stopping old container (if any)...' && \
		docker compose down || true && \
		echo '==> Removing old image (if any)...' && \
		docker rmi $(IMAGE) || true && \
		echo '==> Loading new image...' && \
		gunzip -c $(ARCHIVE) | docker load && \
		echo '==> Starting container...' && \
		docker compose up -d && \
		echo '==> Pruning dangling images...' && \
		docker image prune -f"
	@echo "==> Done. Firewatcher container should be running on $(VM_HOST)."

# Tail container logs
logs:
	ssh $(VM_USER)@$(VM_HOST) "docker logs -f firewatcher"

# Cleanup
clean:
	-docker rmi $(IMAGE)
	-rm -f $(ARCHIVE)

fmt:
	black . && isort .

serve:
	uvicorn server:app --host 0.0.0.0 --port 8000 --reload
