"""
Xcode Intelligence Proxy - Anthropic ↔ OpenAI 格式代理

将 Xcode Intelligence 发出的 Anthropic Messages API 请求，
转换为 OpenAI Chat Completions 格式转发到多种 LLM 后端，
再将响应转换回 Anthropic 格式返回。

用法:
    python main.py
"""

import json
import logging
import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from config import OPENAI_API_URL, OPENAI_API_KEY, SERVER_PORT, SERVER_HOST
from converter import convert_request, convert_response
from streaming import convert_stream

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("xcode-proxy")

app = FastAPI(
    title="Xcode Intelligence Proxy",
    description="Anthropic Messages API → OpenAI Chat Completions 代理",
    version="1.0.0",
)


# ============================================================
#  请求日志中间件：打印所有请求路径/关键头
# ============================================================

class RequestLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 只打关键字段，避免日志过长
        logger.info(
            "REQ %s %s x-api-key=%s anthropic-version=%s user-agent=%s",
            request.method,
            request.url.path,
            request.headers.get("x-api-key", ""),
            request.headers.get("anthropic-version", ""),
            request.headers.get("user-agent", ""),
        )
        return await call_next(request)


app.add_middleware(RequestLogMiddleware)


# ============================================================
#  404 统一包装：避免 Xcode 解析 FastAPI 默认 {detail:"Not Found"} 失败
# ============================================================

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "type": "error",
            "error": {
                "type": "not_found_error",
                "message": f"Not Found: {request.url.path}",
            },
        },
    )



# ============================================================
#  健康检查
# ============================================================

@app.get("/")
async def health_check():
    return {
        "status": "ok",
        "service": "Xcode Intelligence Proxy",
        "target": OPENAI_API_URL,
    }


# ============================================================
#  模型列表端点: GET /v1/models  (Xcode Intelligence 初始化时调用)
# ============================================================

# Xcode 需要拿到 Anthropic 官方格式的模型列表才能正常初始化
# 注意：这里返回的是“给客户端展示/选择的模型列表”。
# 本项目支持的部分后端/路由能力（例如 kimi-code、某些企业内网后端）
# 可能不会在此列表中暴露，详见 README。
_AVAILABLE_MODELS = [
    {
        "id": "openai-default",
        "display_name": "Default (OpenAI-compatible)",
        "created_at": "2025-01-01T00:00:00Z",
        "type": "model",
    },
    {
        "id": "kimi-2.5",
        "display_name": "Kimi 2.5",
        "created_at": "2025-01-01T00:00:00Z",
        "type": "model",
    },
    {
        "id": "minimax-2.5",
        "display_name": "MiniMax 2.5",
        "created_at": "2025-01-01T00:00:00Z",
        "type": "model",
    },
    {
        "id": "glm-4.7",
        "display_name": "GLM-4.7",
        "created_at": "2024-01-01T00:00:00Z",
        "type": "model",
    },
]


@app.get("/v1/models")
async def list_models():
    """返回 Anthropic 官方格式的模型列表"""
    logger.info("收到模型列表请求: GET /v1/models")
    return JSONResponse(content={
        "data": _AVAILABLE_MODELS,
        "has_more": False,
        "first_id": _AVAILABLE_MODELS[0]["id"],
        "last_id": _AVAILABLE_MODELS[-1]["id"],
    })


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    """返回单个模型详情"""
    logger.info(f"收到模型详情请求: GET /v1/models/{model_id}")
    for model in _AVAILABLE_MODELS:
        if model["id"] == model_id:
            return JSONResponse(content=model)
    return JSONResponse(
        status_code=404,
        content={
            "type": "error",
            "error": {
                "type": "not_found_error",
                "message": f"model '{model_id}' not found",
 },
        },
    )


# ============================================================
#  主要端点: POST /v1/messages
# ============================================================

@app.post("/v1/messages")
async def messages(request: Request):
    """
    接收 Anthropic Messages API 格式的请求，转换并代理到 OpenAI 兼容后端。
    """
    return await _handle_anthropic_messages(request)


# Xcode 可能会以 OpenAI 兼容格式直接调用 /v1/chat/completions
# 这里做一个兼容：把它转成 Anthropic Messages API 请求后复用同一套逻辑
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _handle_openai_chat_completions(request)


async def _handle_anthropic_messages(request: Request):
    try:
        # 1. 解析请求体
        body = await request.json()
        logger.info(f"收到请求: model={body.get('model')}, stream={body.get('stream', False)}")

        # 保存原始模型名用于响应
        original_model = body.get("model", "")

        # 2. 转换请求格式: Anthropic → OpenAI
        openai_request = convert_request(body)
        logger.info(f"转换后模型: {openai_request.get('model')}")

        is_stream = body.get("stream", False)

        # 3. 构建转发请求的 headers
        forward_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        }

        if is_stream:
            # ---- 流式模式 ----
            return await _handle_stream(openai_request, forward_headers, original_model)
        else:
            # ---- 非流式模式 ----
            return await _handle_non_stream(openai_request, forward_headers, original_model)

    except httpx.ConnectError as e:
        logger.error(f"连接目标服务失败: {e}")
        return _error_response(
            "connection_error",
            f"无法连接到后端服务: {OPENAI_API_URL}",
            status_code=502,
        )
    except httpx.TimeoutException as e:
        logger.error(f"请求超时: {e}")
        return _error_response(
            "timeout",
            "后端服务响应超时",
            status_code=504,
        )
    except Exception as e:
        logger.error(f"处理请求时发生错误: {e}", exc_info=True)
        return _error_response(
            "internal_error",
            f"代理服务内部错误: {str(e)}",
            status_code=500,
        )


async def _handle_openai_chat_completions(request: Request):
    """接收 OpenAI Chat Completions 格式请求，并按 model 路由到不同后端。"""
    try:
        body = await request.json()
        requested_model = body.get("model")
        logger.info(
            f"收到 OpenAI /v1/chat/completions 请求: model={requested_model}, stream={body.get('stream', False)}"
        )

        # provider 路由（先预置四类）
        from config import (
            BACKEND_MODEL,
            OPENAI_API_URL,
            OPENAI_API_KEY,
            KIMI_CODE_BASE_URL,
            KIMI_CODE_API_KEY,
            KIMI_CODE_MODEL,
            MOONSHOT_BASE_URL,
            MOONSHOT_API_KEY,
            MOONSHOT_MODEL,
            MINIMAX_BASE_URL,
            MINIMAX_API_KEY,
            MINIMAX_MODEL,
            ZHIPU_BASE_URL,
            ZHIPU_API_KEY,
            ZHIPU_MODEL,
        )

        if requested_model == "kimi-2.5":
            # 默认将 "kimi-2.5" 路由到 Moonshot 通用接口（更适合 Xcode 这类客户端）
            url = f"{MOONSHOT_BASE_URL}/v1/chat/completions"
            api_key = MOONSHOT_API_KEY
            body["model"] = MOONSHOT_MODEL
            provider_name = "moonshot"
        elif requested_model == "kimi-code":
            # Kimi Code OpenAI-compatible endpoint（可能会限制只能给特定 Coding Agent 使用）
            url = f"{KIMI_CODE_BASE_URL}/chat/completions"
            api_key = KIMI_CODE_API_KEY
            body["model"] = KIMI_CODE_MODEL
            provider_name = "kimi-code"
        elif requested_model == "minimax-2.5":
            url = f"{MINIMAX_BASE_URL}/v1/chat/completions"
            api_key = MINIMAX_API_KEY
            body["model"] = MINIMAX_MODEL
            provider_name = "minimax"
        elif isinstance(requested_model, str) and requested_model.lower().startswith("glm"):
            # 智谱 GLM（Coding Plan / OpenAI-compatible）：
            # - 端点使用 /chat/completions（而不是 /v1/chat/completions）
            # - 将模型原样透传给后端（例如：GLM-4.7 / glm-5 等）
            url = f"{ZHIPU_BASE_URL}/chat/completions"
            api_key = ZHIPU_API_KEY
            body["model"] = requested_model
            provider_name = "glm"
        elif requested_model == "openai-default":
            # 默认 OpenAI-compatible 后端（由 OPENAI_API_URL / OPENAI_API_KEY / BACKEND_MODEL 决定）
            url = OPENAI_API_URL
            api_key = OPENAI_API_KEY
            body["model"] = BACKEND_MODEL
            provider_name = "default"
        else:
            # 未识别的 model：也回落到默认后端
            url = OPENAI_API_URL
            api_key = OPENAI_API_KEY
            body["model"] = BACKEND_MODEL
            provider_name = "default"

        logger.info(f"转发后端[{provider_name}]: url={url}, model={body.get('model')}, stream={body.get('stream', False)}")

        # OpenAI 兼容鉴权（统一 Bearer；若某家不一致，下次补 provider 级别 auth_header）
        forward_headers = {"Content-Type": "application/json"}
        if api_key:
            forward_headers["Authorization"] = f"Bearer {api_key}"

        # 打印鉴权头是否存在（不泄露完整 key）
        auth_preview = forward_headers.get("Authorization", "")
        if auth_preview:
            logger.info(f"后端[{provider_name}] Authorization: {auth_preview[:16]}... (len={len(auth_preview)})")
        else:
            logger.info(f"后端[{provider_name}] Authorization: <missing>")

        if body.get("stream", False):
            # 将后端 OpenAI SSE 规范化后再返回给 Xcode：
            # - 确保每一条事件是以 `data: {...}\n\n` 为单位输出
            # - 过滤空 chunk/无效 chunk，避免 Xcode 报 Failed to parse SSE
            async def passthrough():
                async with httpx.AsyncClient(timeout=300.0) as client:
                    async with client.stream("POST", url, json=body, headers=forward_headers) as resp:
                        if resp.status_code != 200:
                            data = await resp.aread()
                            logger.error(f"后端[{provider_name}]流式错误: {resp.status_code} - {data.decode(errors='replace')}")
                            return

                        buffer = ""
                        async for raw in resp.aiter_text():
                            buffer += raw
                            while "\n\n" in buffer:
                                event, buffer = buffer.split("\n\n", 1)
                                event = event.strip("\n")
                                if not event:
                                    continue

                                # 只保留以 data: 开头的行
                                if event.startswith("data:"):
                                    # 有些实现会返回 data: {...} 但 delta.content 为空且 role 也为空，Xcode 解析会报错
                                    payload = event[5:].strip()
                                    if payload == "[DONE]":
                                        yield "data: [DONE]\n\n"
                                        continue

                                    try:
                                        obj = json.loads(payload)
                                    except Exception:
                                        # 非 JSON 就原样输出
                                        yield event + "\n\n"
                                        continue

                                    # 规范化：Xcode 似乎不接受 id=null 的 chunk
                                    if obj.get("id") is None:
                                        obj["id"] = "chatcmpl_proxy"

                                    # 规范化：如果 delta.role 为空，补上 assistant（很多客户端要求首包带 role）
                                    choices = obj.get("choices") or []
                                    if choices:
                                        delta = choices[0].get("delta") or {}
                                        if delta.get("role") in (None, ""):
                                            # 仅在第一段/任何段补 role 都可接受
                                            delta["role"] = "assistant"

                                        content = delta.get("content")
                                        finish = choices[0].get("finish_reason")

                                        # 过滤“空 delta”chunk（没有内容也没有结束信号）
                                        is_empty_delta = (content in (None, "") and finish is None)
                                        if is_empty_delta:
                                            continue

                                    yield "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"
                                else:
                                    # 兼容偶发的非 data 事件
                                    yield event + "\n\n"

            return StreamingResponse(
                passthrough(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(url, json=body, headers=forward_headers)
            return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type", "application/json"))

    except Exception as e:
        logger.error(f"处理 OpenAI chat/completions 时发生错误: {e}", exc_info=True)
        return _error_response("internal_error", str(e), status_code=500)


# ============================================================
#  非流式处理
# ============================================================

async def _handle_non_stream(
    openai_request: dict,
    headers: dict,
    original_model: str,
) -> JSONResponse:
    """处理非流式请求"""
    url = OPENAI_API_URL

    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(url, json=openai_request, headers=headers)

        if resp.status_code != 200:
            logger.error(f"后端返回错误: {resp.status_code} - {resp.text}")
            return _error_response(
                "api_error",
                f"后端服务返回错误: {resp.status_code}",
                status_code=resp.status_code,
            )

        openai_response = resp.json()

    # 转换响应格式: OpenAI → Anthropic
    anthropic_response = convert_response(openai_response, original_model)
    logger.info(f"响应成功: stop_reason={anthropic_response.get('stop_reason')}")

    return JSONResponse(content=anthropic_response)


# ============================================================
#  流式处理
# ============================================================

async def _handle_stream(
    openai_request: dict,
    headers: dict,
    original_model: str,
) -> StreamingResponse:
    """处理流式请求"""
    url = OPENAI_API_URL

    async def event_generator():
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST",
                url,
                json=openai_request,
                headers=headers,
            ) as resp:
                if resp.status_code != 200:
                    error_body = await resp.aread()
                    logger.error(f"后端流式返回错误: {resp.status_code} - {error_body.decode()}")
                    # 以 Anthropic error 事件格式返回
                    error_event = {
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": f"后端服务返回错误: {resp.status_code}",
                        },
                    }
                    yield f"event: error\ndata: {json.dumps(error_event, ensure_ascii=False)}\n\n"
                    return

                # 将 OpenAI SSE 流转换为 Anthropic SSE 流
                async for anthropic_event in convert_stream(
                    resp.aiter_bytes(),
                    original_model=original_model,
                ):
                    yield anthropic_event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
        },
    )


# ============================================================
#  错误响应 (Anthropic 格式)
# ============================================================

def _error_response(error_type: str, message: str, status_code: int = 400) -> JSONResponse:
    """返回 Anthropic 格式的错误响应"""
    return JSONResponse(
        status_code=status_code,
        content={
            "type": "error",
            "error": {
                "type": error_type,
                "message": message,
            },
        },
    )


# ============================================================
#  启动
# ============================================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Xcode Intelligence Proxy 启动中...")
    logger.info(f"监听地址: {SERVER_HOST}:{SERVER_PORT}")
    logger.info(f"目标后端: {OPENAI_API_URL}")
    logger.info("=" * 60)

    uvicorn.run(
        "main:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=False,
        log_level="info",
    )
