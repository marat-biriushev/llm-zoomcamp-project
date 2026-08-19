# Shared by the `app` and `ingestion` services — they run different commands over
# the same code and the same dependencies.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.19 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/usr/local

WORKDIR /app

# Dependencies first, in their own layer: they only change when the lock file does,
# so editing application code does not trigger a re-install.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev

# Bake the embedding model into the image. Otherwise every container start would
# download ~90 MB from Hugging Face before it could answer anything — and would fail
# outright without internet access.
ENV HF_HOME=/opt/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')"

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
