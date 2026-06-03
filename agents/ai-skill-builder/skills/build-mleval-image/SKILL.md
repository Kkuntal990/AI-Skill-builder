---
name: build-mleval-image
description: Rebuild and push the mleval-agent Docker image on the remote build host amusing.ucsd.edu, picking up local code changes that live INSIDE the image (sidecar patches, Dockerfile, requirements.txt, entrypoint.sh, analyzer modules). Use whenever a committed change to infra/agents/<name>/{mlevolve_sidecar,Dockerfile,requirements.txt,entrypoint.sh,run_*.py} or src/mleval/analyzer/* must reach the next trajectory pod. Do NOT use for task data / skill markdown (that is PVC-synced via refresh-mleval-pvc, not baked into the image).
---

# Build + push the mleval-agent image (on amusing)

Trajectory pods run a Docker image, NOT the local checkout. Any change to code
that lives *inside* the image is invisible to a new Job until the image is
rebuilt and pushed. This skill encodes the build so it is reproducible and you
never ship a stale image.

## When to invoke

Rebuild the image after committing changes to any of:

- `infra/agents/<name>/mlevolve_sidecar/` (monkey-patches: seed, prompt_logger, skill_retriever, …)
- `infra/agents/<name>/Dockerfile` or `requirements.txt` (dep universe)
- `infra/agents/<name>/entrypoint.sh` (trajectory lifecycle / trap handling)
- `infra/agents/<name>/run_*.py` (sidecar loader shims)
- `src/mleval/analyzer/*` (adapter, stage_classifier, stage_metrics, metrics, …) — these are
  pip-installed into the image, so a runtime analyzer change needs a rebuild

Do **NOT** rebuild for changes to: `infra/tasks/*` data, `infra/skills/*` markdown,
`config.yaml`, or any runtime-mounted file — those are PVC-synced via the
**refresh-mleval-pvc** skill, not baked into the image.

## Why amusing, not this Mac

- amusing is a 32-core Linux amd64 box with native BuildKit (`docker-buildx` v0.18.0). Build ~5–10 min cold, ~1 min cache-hot.
- A Mac build goes through QEMU amd64 emulation: ~30 min. **Do not build locally unless amusing is down.**
- ghcr.io login (write:packages PAT) is already configured on amusing — never re-login or print tokens.

## Prerequisites

- The change must be **committed and pushed** to its branch first. amusing builds from the remote, not your working tree.
- SSH: host `amusing.ucsd.edu`, user `ad-kkokate` (NOT `kuntalkokate`). Key passphrase is the literal string `amusing`.
- `.env` exists on amusing under `~/AI-Skill-builder/.env` (provides `IMAGE`, registry, AIDE/MLEvolve pins). Never echo its values.

## Workflow

1. **Locally**: commit + push the in-image change to its branch (e.g. `mlevolve-smoke`).
   Capture the short SHA — you will verify amusing built exactly this.

2. **Run the build** (one SSH session). Either invoke the helper script:

   ```
   bash scripts/build_on_amusing.sh <branch> [<expected-sha>]
   ```

   …or run the equivalent inline:

   ```
   ssh ad-kkokate@amusing.ucsd.edu '
     cd ~/AI-Skill-builder &&
     git fetch origin &&
     git checkout <branch> &&
     git pull origin <branch> &&
     git submodule update --init --recursive infra/agents/mlevolve/upstream &&
     git log -1 --oneline &&
     set -a && source .env && set +a &&
     make docker-mlevolve &&
     make docker-mlevolve-push
   '
   ```

3. **Build-time safety net**: the Dockerfile runs `infra/agents/<name>/_smoke_imports.py`
   as a build step. If a sidecar refactor broke a patch target or a dual-bind,
   the build FAILS there with an assertion — capture the traceback and fix the
   code; do not treat a failed build as success.

4. **Verify** before trusting the image:
   - amusing's `git log -1 --oneline` shows the expected SHA (the fix is actually in the build).
   - `make docker-mlevolve-push` printed a `latest: digest: sha256:…` line.

## After the build

The image tag (`ghcr.io/kkuntal990/mleval-agent:dev` family) overwrites on every push.
Trajectory pods pull it fresh on next launch, BUT:

- A long-lived **helper pod** caches the old image — redeploy it (`make k8s-apply-helper`, free to do anytime) if you need the new code there for a smoke check.
- This skill does **not** touch the PVC or launch any Job. If the same change set
  also edited task/skill data, run **refresh-mleval-pvc** as well.

## Hard rule

This skill is build + push only. It never applies a trajectory Job to Nautilus —
that still requires explicit user approval per the project's live-run gate.
