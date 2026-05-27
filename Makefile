# ============================================================================
# mleval — Makefile
#
# Sources `.env` automatically (if present) and exports every variable into
# subshells so envsubst / kubectl see them. Override any var inline:
#   `make IMAGE_TAG=v0.1.0 k8s-apply-helper`
#
# Print the effective config at any time:  `make config`
#
# Container build/push targets are intentionally absent — they live in the
# per-agent plugin under `infra/agents/<name>/` (added per agent, not here).
# ============================================================================

# `-include` is the silent variant: no error if .env doesn't exist yet.
-include .env
export

# Sensible defaults so non-deploy targets work without .env. .env overrides
# these via `include` above.
IMAGE_REGISTRY ?= ghcr.io/kkuntal990
IMAGE_NAME     ?= mleval-agent-mlevolve
IMAGE_TAG      ?= dev
GPU_TYPE       ?= nvidia.com/gpu
GPU_PRODUCT    ?= NVIDIA-A10
IMAGE          := $(IMAGE_REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)

.PHONY: install fmt lint typecheck test check clean \
        docker-mlevolve docker-mlevolve-push \
        config _require_env _require_ns _require_api_key _require_ghcr_token \
        k8s-secret k8s-ghcr-pull-secret k8s-apply-pvc \
        k8s-apply-helper k8s-delete-helper \
        ab-plan ab-apply ab-wait \
        analyze-trajectory aggregate-run \
        mlevolve-submodule-init

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
	@echo "GPU_PRODUCT       = $(GPU_PRODUCT)"
	@echo "MLEVAL_LLM_MODEL  = $${MLEVAL_LLM_MODEL:-<unset>}"
	@echo "MLEVAL_RUN_ID     = $${MLEVAL_RUN_ID:-<unset>}"
	@echo "DEFAULT_SEED      = $${DEFAULT_SEED:-<unset>}"
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

_require_ghcr_token: _require_env
	@test -n "$$GHCR_READ_TOKEN" -a "$$GHCR_READ_TOKEN" != "REPLACE_ME" \
	    || { echo "ERROR: GHCR_READ_TOKEN not set in .env (need a read:packages PAT)" >&2; exit 1; }

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

# Single-agent on this branch: MLEvolve. AIDE removed during the
# mlevolve-smoke spike (see docs/eval/stage2.md).

mlevolve-submodule-init:
	git submodule update --init --recursive infra/agents/mlevolve/upstream

docker-mlevolve: mlevolve-submodule-init
	docker build -f infra/agents/mlevolve/Dockerfile -t $(IMAGE) .

docker-mlevolve-push:
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

# Docker-registry Secret so Nautilus nodes can pull our private ghcr.io image.
# Idempotent. References this Secret via `imagePullSecrets: [{name: ghcr-pull}]`
# in helper/job manifests.
k8s-ghcr-pull-secret: _require_ns _require_ghcr_token
	kubectl -n $$K8S_NAMESPACE delete secret ghcr-pull --ignore-not-found
	kubectl -n $$K8S_NAMESPACE create secret docker-registry ghcr-pull \
	    --docker-server=ghcr.io \
	    --docker-username=kkuntal990 \
	    --docker-password="$$GHCR_READ_TOKEN" \
	    --docker-email=kukokate@ucsd.edu

k8s-apply-helper: _require_ns
	envsubst < deploy/k8s/helper-jupyter-1gpu.yaml | kubectl -n $$K8S_NAMESPACE apply -f -

k8s-delete-helper: _require_ns
	kubectl -n $$K8S_NAMESPACE delete pod mleval-jupyter-1gpu --ignore-not-found

# Pre-download task (and optionally skill) requirements into the PVC pip
# cache so the first trajectory of a sweep doesn't pay the wheel-download
# cost (~30-60s for the tabular stack, more for PEFT). Idempotent. Re-run
# whenever a requirements.txt changes.
#
#     make pip-warm TASK=house-prices
#     make pip-warm TASK=house-prices SKILL=tabular-baseline
pip-warm: _require_ns
	@test -n "$(TASK)" -a "$(TASK)" != "_template" || { echo "ERROR: TASK= is required" >&2; exit 1; }
	@SKILL_REQS=""; \
	if [ -n "$(SKILL)" ]; then SKILL_REQS=/results/skills/$(SKILL)/requirements.txt; fi; \
	TASK_SLUG=$$(echo "$(TASK)" | tr '[:upper:]_' '[:lower:]-' | sed 's/[^a-z0-9-]/-/g'); \
	export TASK_SLUG TASK=$(TASK) IMAGE_REGISTRY=$(IMAGE_REGISTRY) IMAGE_NAME=$(IMAGE_NAME) IMAGE_TAG=$(IMAGE_TAG); \
	export REQS_PATHS="/results/data/$(TASK)/requirements.txt $$SKILL_REQS"; \
	envsubst < deploy/k8s/pip-warm.yaml | kubectl -n $$K8S_NAMESPACE apply -f -; \
	echo "[make] waiting for pip-warm-$$TASK_SLUG pod to complete..."; \
	kubectl -n $$K8S_NAMESPACE wait --for=condition=Ready pod/pip-warm-$$TASK_SLUG --timeout=120s; \
	kubectl -n $$K8S_NAMESPACE logs -f pod/pip-warm-$$TASK_SLUG; \
	kubectl -n $$K8S_NAMESPACE delete pod pip-warm-$$TASK_SLUG --wait=false

# ---- A/B sweep orchestration --------------------------------------------
#
# `ab-plan` previews trajectories without touching the cluster.
# `ab-apply` actually applies the Jobs; `ab-wait` polls until completion.
# Drive task/seed/skill-path via TASK, SEEDS, SKILL_PATH variables; e.g.:
#     make ab-plan  TASK=mytask SEEDS="0 1" SKILL_PATH=/results/skills/peft/SKILL.md
#     make ab-apply TASK=mytask SEEDS="0 1" SKILL_PATH=/results/skills/peft/SKILL.md

TASK         ?= _template
SEEDS        ?= 0 1
SKILL_PATH   ?=
TIME_LIMIT   ?= 1800
STEP_LIMIT   ?= 5
PROFILE      ?= gpu
LLM_TIMEOUT  ?= 120

ab-plan: _require_ns
	python3 -m infra.orchestrator.run_ab \
	    --task $(TASK) --seeds $(SEEDS) \
	    --skill-path "$(SKILL_PATH)" \
	    --time-limit-sec $(TIME_LIMIT) --step-limit $(STEP_LIMIT) \
	    --llm-timeout-sec $(LLM_TIMEOUT) \
	    --profile $(PROFILE)

ab-apply: _require_ns _require_api_key
	python3 -m infra.orchestrator.run_ab \
	    --task $(TASK) --seeds $(SEEDS) \
	    --skill-path "$(SKILL_PATH)" \
	    --time-limit-sec $(TIME_LIMIT) --step-limit $(STEP_LIMIT) \
	    --llm-timeout-sec $(LLM_TIMEOUT) \
	    --profile $(PROFILE) \
	    --apply

ab-wait: _require_ns _require_api_key
	python3 -m infra.orchestrator.run_ab \
	    --task $(TASK) --seeds $(SEEDS) \
	    --skill-path "$(SKILL_PATH)" \
	    --time-limit-sec $(TIME_LIMIT) --step-limit $(STEP_LIMIT) \
	    --llm-timeout-sec $(LLM_TIMEOUT) \
	    --profile $(PROFILE) \
	    --apply --wait

# ---- post-trajectory analyzers (local invocation; pod-side runs in entrypoint)
#
# Useful when iterating on the analyzer code against a pulled trajectory:
#     make analyze-trajectory DIR=./pulled-results/mvp-001/trajectory_id_xyz

analyze-trajectory:
	@test -n "$(DIR)" || { echo "ERROR: pass DIR=./path/to/trajectory" >&2; exit 1; }
	python3 -m mleval.analyzer.adapter_mlevolve $(DIR)
	python3 -m mleval.analyzer.stage_classifier $(DIR)
	python3 -m mleval.analyzer.state_predicates $(DIR)

aggregate-run:
	@test -n "$(RUN_DIR)" || { echo "ERROR: pass RUN_DIR=./path/to/run-root" >&2; exit 1; }
	python3 -m mleval.analyzer.aggregate $(RUN_DIR)

# ---- cleanup -------------------------------------------------------------

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
