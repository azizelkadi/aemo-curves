# syntax=docker/dockerfile:1
# Slim dev image for the aemo-curves project.
#   - Python 3.11 (matches .python-version and requires-python >=3.11,<3.13)
#   - uv for dependency management (copied from the official image, pinned)
#   - Node 22 + Claude Code CLI
#   - build tools so new C-extension wheels can still be built inside the box
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

# uv / uvx binaries from the official image (pinned to the host's version).
COPY --from=ghcr.io/astral-sh/uv:0.7.6 /uv /uvx /usr/local/bin/

# System packages: git, build tools, and the bits NodeSource needs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        gnupg \
        sudo \
        build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g @anthropic-ai/claude-code \
    && npm cache clean --force \
    && rm -rf /var/lib/apt/lists/*

# Non-root user with passwordless sudo (needed to claim the venv volume).
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME \
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && mkdir -p /opt/venv \
    && chown -R $USER_UID:$USER_GID /opt/venv

USER $USERNAME
