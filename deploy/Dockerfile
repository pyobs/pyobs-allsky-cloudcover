# pyobs-allsky-cloudcover as a container.
#   docker build -f deploy/Dockerfile -t pyobs-cloudcover .
#   docker run -v /etc/pyobs/cloudcover.yaml:/config.yaml:ro -v /var/lib/allsky/images:/images \
#              -p 8080:8080 pyobs-cloudcover
# In the config, set watch.path: /images and server.url: 0.0.0.0.
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv PATH=/opt/venv/bin:$PATH

WORKDIR /app
COPY pyproject.toml uv.lock Readme.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY pyobs_cloudcover ./pyobs_cloudcover
RUN uv sync --frozen --no-dev

RUN useradd --system --create-home pyobs
USER pyobs
EXPOSE 8080
CMD ["pyobs", "/config.yaml"]
