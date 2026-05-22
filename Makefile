AI ?=
LOG := .ci-ai.log

ifdef AI
_goals := $(or $(MAKECMDGOALS),ci)
.PHONY: $(_goals)
$(_goals):
	@rm -f $(LOG)
	@$(MAKE) --no-print-directory AI= $@ > $(LOG) 2>&1 \
		&& echo "✅ $@ passed (log: $(LOG))" \
		|| (echo "❌ $@ failed:"; tail -50 $(LOG); echo "(full log: $(LOG))"; exit 1)

else

# ---------------------------------------------------------------------------
# Peer-repo discovery for dev-local target
# ---------------------------------------------------------------------------
PEER_BOOK_TOOLS_PATH := ../pd-book-tools
PEER_BOOK_TOOLS := $(realpath $(PEER_BOOK_TOOLS_PATH))

define _require_peer_book_tools
	@if [ -z "$(PEER_BOOK_TOOLS)" ]; then \
		echo ""; \
		echo "❌  Cannot find pd-book-tools at $(PEER_BOOK_TOOLS_PATH)"; \
		echo "    Clone it first:  git clone https://github.com/ConcaveTrillion/pd-book-tools.git ../pd-book-tools"; \
		echo ""; \
		exit 1; \
	fi
endef

.PHONY: setup lint lint-check format typecheck test ci build clean pre-commit-check dev-local

setup: ## Install dependencies (idempotent)
	uv sync --group dev
	@HOOKS_PATH="$$(git config core.hooksPath 2>/dev/null || echo '.git/hooks')"; \
	  [ -f "$$HOOKS_PATH/pre-commit" ] || uv run pre-commit install --hook-type pre-commit --hook-type commit-msg

lint: ## Run linting (auto-fix)
	uv run ruff check --select I --fix
	uv run ruff check --fix

lint-check: ## Read-only ruff format+check (no auto-fix; matches CI exactly)
	uv run ruff format --check .
	uv run ruff check .

format: ## Format code
	uv run ruff format pd_ocr_training tests

typecheck: ## Run basedpyright at recommended mode (workspace canonical)
	uv run basedpyright pd_ocr_training --level error

test: ## Run tests with parallelization
	uv run pytest -n auto

pre-commit-check: ## Run all pre-commit hooks against all files (read-only check)
	uv run pre-commit run --all-files

ci: ## Run complete CI pipeline (setup, pre-commit, lint-check, typecheck, test)
	@$(MAKE) --no-print-directory setup
	@$(MAKE) --no-print-directory pre-commit-check
	@$(MAKE) --no-print-directory lint-check
	@$(MAKE) --no-print-directory typecheck
	@$(MAKE) --no-print-directory test

build: ## Build the project
	uv build

dev-local: ## [local-dev] Install pd-book-tools from ../pd-book-tools as editable in the venv
	$(call _require_peer_book_tools)
	@echo "Installing pd-book-tools editable from $(PEER_BOOK_TOOLS)..."
	UV_LINK_MODE=copy uv pip install -e "$(PEER_BOOK_TOOLS)"
	UV_LINK_MODE=copy uv pip install -e . --no-deps
	UV_LINK_MODE=copy uv pip install --group dev
	@touch .venv/.pd-dev-local
	@echo "Local editable pd-book-tools is active in the venv."

clean: ## Clean cache and temporary files
	rm -rf dist .venv .pytest_cache .ruff_cache .ci-ai.log htmlcov

endif
