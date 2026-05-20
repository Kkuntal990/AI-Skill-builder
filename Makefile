# ============================================================================
# mleval — Makefile
#
# Sources `.env` automatically (if present) and exports every variable into
# subshells so envsubst / kubectl / docker see them. Override any var
# inline:   `make IMAGE_TAG=v0.1.0 docker-agent`
#
# Print the effective config at any time:  `make config`
# ============================================================================

# `-include` is the silent variant: no error if .env doesn't exist yet.
-include .env
export

# Sensible defaults so non-deploy targets work without .env. .env overrides
# these via `include` above.
IMAGE_REGISTRY ?= ghcr.io/kkuntal990
IMAGE_NAME     ?= mleval-agent
IMAGE_TAG      ?= dev
GPU_TYPE       ?= nvidia.com/rtxa6000
MLEVOLVE_REPO  ?= https://github.com/InternScience/MLEvolve.git
MLEVOLVE_REF   ?= main
IMAGE          := $(IMAGE_REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)

.PHONY: install fmt lint typecheck test check clean \
        docker-agent docker-push \
        config _require_env _require_ns _require_api_key \
        k8s-secret k8s-apply-pvc \
        k8s-apply-helper k8s-delete-helper \
        k8s-apply-job-debug-trl-grpo k8s-delete-job-debug-trl-grpo

# ---- environment ---------------------------------------------------------

install:
	pip install -e ".[dev]"
	pre-commit install || true

# Show what config would actually be used by a deploy right now. The user
# asked for `.env` to be the "ground truth" — this target is how to check it.
config:
	@echo "IMAGE             = $(IMAGE)"
	@echo "K8S_NAMESPACE     = $${K8S_NAMESPACE:-<unset>}"
	@echo "GPU_TYPE          = $(GPU_TYPE)"
	@echo "MLEVAL_LLM_MODEL  = $${MLEVAL_LLM_MODEL:-<unset>}"
	@echo "MLEVAL_RUN_ID     = $${MLEVAL_RUN_ID:-<unset>}"
	@echo "DEFAULT_SEED      = $${DEFAULT_SEED:-<unset>}"
	@echo "MLEVOLVE_REPO     = $(MLEVOLVE_REPO)"
	@echo "MLEVOLVE_REF      = $(MLEVOLVE_REF)"
	@if [ -n "$$OPENROUTER_API_KEY" ] && [ "$$OPENROUTER_API_KEY" != "REPLACE_ME" ]; then \
	    echo "OPENROUTER_API_KEY = <set>"; \
	 elif [ "$$OPENROUTER_API_KEY" = "REPLACE_ME" ]; then \
	    echo "OPENROUTER_API_KEY = <REPLACE_ME — edit .env>"; \
	 else echo "OPENROUTER_API_KEY = <unset>"; fi
	@if [ -n "$$HF_TOKEN" ] && [ "$$HF_TOKEN" != "REPLACE_ME" ]; then \
	    echo "HF_TOKEN          = <set>"; \
	 else echo "HF_TOKEN          = <unset>"; fi

# ---- guards (internal) ---------------------------------------------------

_require_env:
	@test -f .env || { echo "ERROR: .env missing. Run: cp .env.example .env && \$$EDITOR .env" >&2; exit 1; }

_require_ns: _require_env
	@test -n "$$K8S_NAMESPACE" -a "$$K8S_NAMESPACE" != "REPLACE_ME" \
	    || { echo "ERROR: K8S_NAMESPACE not set in .env" >&2; exit 1; }

_require_api_key: _require_env
	@test -n "$$OPENROUTER_API_KEY" -a "$$OPENROUTER_API_KEY" != "REPLACE_ME" \
	    || { echo "ERROR: OPENROUTER_API_KEY not set in .env" >&2; exit 1; }

# ---- code quality --------------------------------------------------------

fmt:
	ruff format src tests
	ruff check --fix src tests

lint:
	ruff check src tests
	ruff format --check src tests

typecheck:
	mypy src

test:
	pytest -q

check: lint typecheck test

# ---- containers ----------------------------------------------------------

docker-agent:
	docker build \
	    --build-arg MLEVOLVE_REPO=$(MLEVOLVE_REPO) \
	    --build-arg MLEVOLVE_REF=$(MLEVOLVE_REF) \
	    -f docker/agent.Dockerfile -t $(IMAGE) .

docker-push:
	docker push $(IMAGE)

# ---- kubernetes ----------------------------------------------------------
#
# All targets that touch the cluster pre-flight on `_require_ns` (and
# `_require_api_key` for secret creation). YAMLs are rendered with
# `envsubst` so the live config matches what `make config` reports.

k8s-apply-pvc: _require_ns
	kubectl -n $$K8S_NAMESPACE apply -f deploy/k8s/pvc.yaml

# Provision (or refresh) the mleval-secrets Secret from `.env`. Idempotent.
k8s-secret: _require_ns _require_api_key
	kubectl -n $$K8S_NAMESPACE delete secret mleval-secrets --ignore-not-found
	kubectl -n $$K8S_NAMESPACE create secret generic mleval-secrets \
	    --from-literal=openrouter-api-key="$$OPENROUTER_API_KEY" \
	    --from-literal=hf-token="$${HF_TOKEN:-}"

k8s-apply-helper: _require_ns
	envsubst < deploy/k8s/helper-jupyter-1gpu.yaml | kubectl -n $$K8S_NAMESPACE apply -f -

k8s-delete-helper: _require_ns
	kubectl -n $$K8S_NAMESPACE delete pod mleval-jupyter-1gpu --ignore-not-found

k8s-apply-job-debug-trl-grpo: _require_ns
	envsubst < deploy/k8s/job-debug-trl-grpo.yaml | kubectl -n $$K8S_NAMESPACE apply -f -

k8s-delete-job-debug-trl-grpo: _require_ns
	kubectl -n $$K8S_NAMESPACE delete job mleval-debug-trl-grpo-mvp --ignore-not-found

# ---- cleanup -------------------------------------------------------------

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
