#!/usr/bin/env bash
# Build + push the mleval-agent image on the remote build host amusing.ucsd.edu.
#
# Usage:
#   build_on_amusing.sh <branch> [<expected-sha>]
#
#   <branch>        git branch to build (e.g. mlevolve-smoke)
#   <expected-sha>  optional short SHA to assert amusing checked out before building
#
# Why a remote build: amusing is native amd64 with BuildKit (~5-10 min cold).
# A Mac build is QEMU-emulated (~30 min). Build here unless amusing is down.
#
# ghcr.io login is already configured on amusing — this script never touches
# credentials. .env on amusing is the source of truth for the image tag.
#
# SSH: user ad-kkokate (NOT kuntalkokate). Key passphrase is the literal
# string `amusing` if prompted.
set -euo pipefail

BRANCH="${1:?usage: build_on_amusing.sh <branch> [<expected-sha>]}"
EXPECTED_SHA="${2:-}"
HOST="ad-kkokate@amusing.ucsd.edu"

echo "[build] branch=${BRANCH} expected-sha=${EXPECTED_SHA:-<any>} host=${HOST}"

# Single SSH session: sync, verify SHA, source env, build (runs _smoke_imports.py
# as a Docker build step), push. Any failure aborts via set -e on the remote.
ssh "${HOST}" "
  set -euo pipefail
  cd ~/AI-Skill-builder
  git fetch origin
  git checkout '${BRANCH}'
  git pull origin '${BRANCH}'
  git submodule update --init --recursive infra/agents/mlevolve/upstream
  HEAD_SHA=\$(git rev-parse --short HEAD)
  echo \"[build] amusing HEAD = \$(git log -1 --oneline)\"
  if [ -n '${EXPECTED_SHA}' ] && ! git merge-base --is-ancestor '${EXPECTED_SHA}' HEAD; then
    echo \"[build] ERROR: expected sha ${EXPECTED_SHA} is not in HEAD \$HEAD_SHA\" >&2
    exit 3
  fi
  set -a && source .env && set +a
  make docker-mlevolve     # Dockerfile runs _smoke_imports.py — fails build on patch regression
  make docker-mlevolve-push
  echo '[build] done — image pushed'
"
