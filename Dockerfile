FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e . 2>/dev/null || true

# Copy source
COPY src/ src/
RUN pip install --no-cache-dir -e .

EXPOSE 8002

CMD ["uvicorn", "paperly.app:app", "--host", "0.0.0.0", "--port", "8002"]
