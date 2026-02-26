# Xcode Intelligence Proxy

把 **Xcode Intelligence** 发出的 **Anthropic Messages API** 请求转发到 **OpenAI-compatible** 后端，并将响应转换回 Anthropic 协议返回。
另外也支持 Kimi、minimax、glm

适用于：
- 让 Xcode / 其他 Anthropic 客户端接入任意 OpenAI-compatible LLM 后端
- 在同一个代理里按 `model` 做多后端路由（可选）
- xcode配置http://localhost:5588，其他随便填写即可



## Features

- 兼容 Anthropic Messages API：`POST /v1/messages`
- 兼容 OpenAI Chat Completions：`POST /v1/chat/completions`
- SSE 流式转发/转换（对 Xcode 解析更友好）
- 多后端路由（按 `model`）：Moonshot / MiniMax / GLM（示例）

## Endpoints

- `GET /`：健康检查
- `GET /v1/models`：模型列表（给 Xcode 初始化用）
- `POST /v1/messages`：Anthropic Messages API
- `POST /v1/chat/completions`：OpenAI Chat Completions（会按 `model` 路由到不同后端）

## Model list 说明（重要）

`GET /v1/models` 返回的是 **给客户端展示/选择的模型列表**。

当前行为：
- **Kimi Code**：当前 **不会** 在 `GET /v1/models` 中返回（但代码层面仍保留 `model=kimi-code` 的路由能力）。
- **默认后端（OPENAI_API_URL / OPENAI_API_KEY / BACKEND_MODEL）**：通过 `openai-default` 在 `GET /v1/models` 中暴露为一个可选项。

如需暴露/隐藏更多模型，请按需修改 `main.py` 的 `_AVAILABLE_MODELS`。

## Configuration

推荐使用 `.env`（不要提交到 GitHub）。可以从 `.env.example` 拷贝：

```bash
cp .env.example .env
```

关键变量：

- `OPENAI_API_URL`：默认 OpenAI-compatible 后端的完整 URL（直接到 `/v1/chat/completions`）
- `OPENAI_API_KEY`：默认后端 key
- `BACKEND_MODEL`：默认后端模型名

多后端（可选）：

- Moonshot（Xcode 里可用 `model=kimi-2.5` 触发路由）
  - `MOONSHOT_BASE_URL` / `MOONSHOT_API_KEY` / `MOONSHOT_MODEL`
- MiniMax（`model=minimax-2.5`）
  - `MINIMAX_BASE_URL` / `MINIMAX_API_KEY` / `MINIMAX_MODEL`
- GLM / 智谱（`model` 以 `glm` 开头，例如 `GLM-4.7` / `glm-5`）
  - `ZHIPU_BASE_URL` / `ZHIPU_API_KEY`
  - 说明：当前实现会请求 `${ZHIPU_BASE_URL}/chat/completions`（适配 BigModel Coding Plan 的 OpenAI-compatible 网关），并将 `model` 原样透传给后端

> 提示：所有 API Key 都应通过环境变量提供，项目代码与 `.env.example` 中不包含任何真实密钥。

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

默认监听：`0.0.0.0:5588`

## Docker

```bash
docker build -t xcode-intelligence-proxy .
docker run --rm -p 5588:5588 --env-file .env xcode-intelligence-proxy
```

## Docker Compose

```bash
docker compose up -d --build
curl http://localhost:5588/
```

## Podman

macOS 上通常需要先启动 podman machine：

```bash
podman machine start
podman build -t xcode-intelligence-proxy .
podman run -d --name xcode-intelligence-proxy --env-file .env -p 5588:5588 xcode-intelligence-proxy
```

查看日志：

```bash
podman logs -f xcode-intelligence-proxy
```

## Security notes（开源前检查清单）

- 确保 `.env` **不提交**（仓库已在 `.gitignore` 中忽略 `.env`）
- 如曾经把真实 key 写入过文件，请在推送前：
  - 检查 `git diff` / `git log -p` 是否包含密钥
  - 必要时 **更换/吊销** 已泄露的 key（因为 git 历史可能被公开）
- 建议在 GitHub 仓库开启 secret scanning

## License

MIT
