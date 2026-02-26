# ============================================================
#  Xcode Intelligence Proxy - Dockerfile
#  轻量级 Python 镜像，用于容器化部署 API 代理服务
# ============================================================

FROM python:3.12-slim

# 防止 Python 写 .pyc 文件 & 立即刷新日志到 stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 先复制依赖清单并安装（利用 Docker 层缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY main.py config.py converter.py streaming.py ./

# 暴露默认服务端口
EXPOSE 5588

# 启动服务
CMD ["python", "main.py"]
