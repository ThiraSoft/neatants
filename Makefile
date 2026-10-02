# NeatAnts: the game (window) and the headless evolution runner.

GO ?= go

.DEFAULT_GOAL := help
.PHONY: help build game headless test test-long run evolve clean clean-saves

help: ## List the available commands
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  make %-12s %s\n", $$1, $$2}'

build: game headless ## Build the game and the headless binary

game: ## Build the game (neatants)
	$(GO) build -o neatants ./cmd/neatants

headless: ## Build the windowless binary (neatants-headless)
	$(GO) build -o neatants-headless ./cmd/neatants-headless

test: ## Vet the code and run the fast tests
	$(GO) vet ./...
	$(GO) test ./...

test-long: ## Also run the long tests (diversity, marriage, learning...)
	NEATANTS_LONG_TESTS=1 $(GO) test -timeout 120m -v ./...

run: game ## Build then launch the game
	./neatants

evolve: headless ## Evolve without a window, one world per core (WORLDS=1 for a single one) · Ctrl+C to stop
	./neatants-headless $(if $(WORLDS),-worlds $(WORLDS))

clean: ## Remove both binaries
	rm -f neatants neatants-headless

clean-saves: ## Empty the saves directory (start from scratch)
	rm -rf saves/*
