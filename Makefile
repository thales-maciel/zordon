IMAGE ?= ghcr.io/thales-maciel/zordon:latest
PORT ?= 8080

.PHONY: help test build check run sync-data preprocess compose-up compose-down smoke bench image submission clean fmt

help:
	@printf '%s\n' \
		'Targets:' \
		'  make test          Run Zig unit tests' \
		'  make build         Build the release binary' \
		'  make check         Run tests and build' \
		'  make run           Run the API locally on PORT=8080' \
		'  make sync-data     Copy official resources from the contest checkout' \
		'  make preprocess    Build data/model/references.i16.bin' \
		'  make compose-up    Start the contest topology locally' \
		'  make compose-down  Stop the local topology' \
		'  make smoke         Run the contest smoke test via Docker Compose' \
		'  make bench         Run the contest k6 benchmark via Docker Compose' \
		'  make image         Build the linux/amd64 image' \
		'  make submission    Refresh the submission worktree' \
		'  make clean         Remove Zig build outputs'

test:
	zig build test

build:
	zig build --release=fast

check: test build

run:
	PORT=$(PORT) zig build run

sync-data:
	./scripts/sync-contest-data.sh

preprocess:
	./scripts/preprocess.sh

compose-up:
	docker compose up --build

compose-down:
	docker compose down --remove-orphans

smoke:
	./scripts/smoke.sh

bench:
	./scripts/bench.sh

image:
	IMAGE=$(IMAGE) ./scripts/build-image.sh

submission:
	./scripts/make-submission-branch.sh

fmt:
	zig fmt build.zig src tools

clean:
	rm -rf .zig-cache zig-out
