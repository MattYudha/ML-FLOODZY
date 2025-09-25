FROM python:3.11-slim AS base
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

FROM base AS runtime
WORKDIR /app
COPY api ./api
COPY ml ./ml
COPY data ./data
COPY models ./models
COPY reports ./reports
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 CMD wget -qO- http://localhost:8000/healthz || exit 1
CMD ["uvicorn","api.main:app","--host","0.0.0.0","--port","8000"]
