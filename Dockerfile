FROM python:3.12-slim AS core

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home app \
    && mkdir -p /app/data \
    && chown -R app:app /app

COPY --chown=app:app pyproject.toml README.md LICENSE ./
COPY --chown=app:app src ./src

USER app

ENTRYPOINT ["python", "-m", "baby_monitor"]
CMD ["--help"]
