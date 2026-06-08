.PHONY: help install lint format type test test-unit test-integration test-e2e test-llm clean ollama-pull migrate makemigration

help:
	@echo "Serenade dev targets:"
	@echo "  make install           install Python deps via uv (dev extras)"
	@echo "  make lint              ruff check + format"
	@echo "  make format            auto-format with ruff"
	@echo "  make type              mypy strict"
	@echo "  make test              unit tests"
	@echo "  make test-unit         unit tests (explicit)"
	@echo "  make test-integration  integration tests"
	@echo "  make test-e2e          end-to-end tests (requires DVWA target)"
	@echo "  make test-llm          live LLM tests (requires Ollama or Anthropic key)"
	@echo "  make ollama-pull       pull required Ollama models (gemma4:26b + gemma4)"
	@echo "  make migrate           apply alembic migrations"
	@echo "  make makemigration     create a new alembic migration (MSG='...')"
	@echo "  make clean             remove caches and build artifacts"

install:
	uv pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

type:
	mypy .

test: test-unit

test-unit:
	pytest -m "not integration and not sandbox and not llm" -q

test-integration:
	pytest -m integration -q

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov dist build
	find . -type d -name __pycache__ -exec rm -rf {} +

ollama-pull:
	@command -v ollama >/dev/null 2>&1 || { echo "Install Ollama first: curl -fsSL https://ollama.com/install.sh | sh"; exit 1; }
	ollama pull gemma4:26b
	ollama pull gemma4

# Database migrations (alembic)
migrate:
	alembic upgrade head

makemigration:
	@test -n "$(MSG)" || { echo "Usage: make makemigration MSG='describe the change'"; exit 1; }
	alembic revision --autogenerate -m "$(MSG)"

# E2E + live-LLM gated tests
test-e2e:
	@echo "Requires: content/targets/dvwa stack up (see content/targets/dvwa/README.md)"
	RUN_E2E=1 pytest -m integration tests/integration/test_dvwa_e2e.py -v

test-llm:
	@echo "Requires: ANTHROPIC_API_KEY or Ollama running (ollama serve)"
	RUN_LIVE_LLM=1 pytest -m llm tests/integration/test_live_llm.py -v
