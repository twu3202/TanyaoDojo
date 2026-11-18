FROM rust:1.91-bookworm AS native
RUN apt-get update && apt-get install -y --no-install-recommends python3-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /native
COPY native_source/ ./
ENV PYO3_PYTHON=/usr/bin/python3
RUN cargo build --locked --release -p libriichi --lib

FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY --chown=user . .
COPY --from=native --chown=user /native/target/release/libriichi.so /home/user/app/assets/libriichi.so
USER user
ENV HOST=0.0.0.0 PORT=7860 PYTHONUNBUFFERED=1 TANYAO_ASSETS=/home/user/app/assets TANYAO_NATIVE_DIR=/home/user/app/assets TANYAO_TORCH_THREADS=2
RUN python -c "from common.suit_table import load_table; load_table()"
EXPOSE 7860
CMD ["python", "-m", "deploy.play.server"]
