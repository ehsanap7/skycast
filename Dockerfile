FROM python:3.14.7-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --locked --no-dev --extra orchestration --no-install-project

COPY src ./src
RUN uv sync --locked --no-dev --extra orchestration --no-editable

RUN useradd --create-home --uid 1000 skycast \
    && mkdir -p /app/data \
    && chown skycast:skycast /app/data

USER skycast

CMD ["skycast", "data", "schedule", "--config-dir", "/app/config"]
