FROM python:3.11-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    inotify-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# 放在 Dockerfile 里 requirements 安装之前
RUN mkdir -p /root/.pip && \
    echo "[global]\n\
index-url = https://pypi.tuna.tsinghua.edu.cn/simple\n\
trusted-host = pypi.tuna.tsinghua.edu.cn\n" \
    > /root/.pip/pip.conf

    COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY pyproject.toml ./
COPY conf ./conf
COPY src ./src

ENV KEDRO_ENV=base \
    PYTHONPATH=/app/src

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]

