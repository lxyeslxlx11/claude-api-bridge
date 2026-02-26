"""
格式转换器 - Anthropic Messages API ↔ OpenAI Chat Completions API

方向:
  请求: Anthropic → OpenAI  (Xcode 发来的请求 → 转发给个人部署模型)
  响应: OpenAI → Anthropic  (个人部署模型的响应 → 返回给 Xcode)
"""

import time
import uuid
from typing import Any

from config import map_model_name, DEFAULT_MAX_TOKENS


# ============================================================
#  finish_reason / stop_reason 映射
# ============================================================

OPENAI_TO_ANTHROPIC_STOP_REASON = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
    None: None,
}

ANTHROPIC_TO_OPENAI_FINISH_REASON = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    None: None,
}


def map_stop_reason(openai_finish_reason: str | None) -> str | None:
    """OpenAI finish_reason → Anthropic stop_reason"""
    return OPENAI_TO_ANTHROPIC_STOP_REASON.get(
        openai_finish_reason, openai_finish_reason
    )


# ============================================================
#  请求转换: Anthropic → OpenAI
# ============================================================


def _normalize_content(content: Any) -> str:
    """
    将 Anthropic content 格式统一转为 OpenAI 的字符串格式。

    Anthropic 支持:
      - 字符串: "hello"
      - 数组: [{"type": "text", "text": "hello"}, ...]
    OpenAI 只需要字符串。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts) if parts else ""
    return str(content) if content else ""


def convert_request(anthropic_request: dict) -> dict:
    """
    将 Anthropic Messages API 请求转换为 OpenAI Chat Completions 请求。

    Anthropic 请求示例:
    {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "system": "You are a helpful assistant.",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "temperature": 0.7,
        "stream": true
    }

    转换为 OpenAI 请求:
    {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"}
        ],
        "temperature": 0.7,
        "stream": true,
        "stream_options": {"include_usage": true}
    }
    """
    openai_messages = []

    # 1. 提取 system 参数 → 插入为 messages 消息
    system_content = anthropic_request.get("system")
    if system_content:
        # Anthropic system 可以是字符串或 content block 数组
        system_text = _normalize_content(system_content)
        if system_text:
            openai_messages.append({"role": "system", "content": system_text})

    # 2. 转换 messages 数组
    for msg in anthropic_request.get("messages", []):
        role = msg.get("role", "user")
        content = _normalize_content(msg.get("content", ""))
        openai_messages.append({"role": role, "content": content})

    # 3. 构建 OpenAI 请求体
    openai_request = {
        "model": map_model_name(anthropic_request.get("model", "")),
        "messages": openai_messages,
        "max_tokens": anthropic_request.get("max_tokens", DEFAULT_MAX_TOKENS),
    }

    # 可选参数
    if "temperature" in anthropic_request:
        openai_request["temperature"] = anthropic_request["temperature"]

    if "top_p" in anthropic_request:
        openai_request["top_p"] = anthropic_request["top_p"]

    if "stop_sequences" in anthropic_request:
        openai_request["stop"] = anthropic_request["stop_sequences"]

    stream = anthropic_request.get("stream", False)
    openai_request["stream"] = stream

    # 流式请求时让 OpenAI 返回 usage 信息
    if stream:
        openai_request["stream_options"] = {"include_usage": True}

    return openai_request


# ======================================================
#  响应转换: OpenAI → Anthropic (非流式)
# ============================================================


def convert_response(openai_response: dict, original_model: str = "") -> dict:
    """
    将 OpenAI Chat Completions 响应转换为 Anthropic Messages API 响应。

    OpenAI 响应示例:
    {
        "id": "chatcmpl-xxx",
        "object": "chat.completion",
        "created": 1721075651,
        "model": "claude-sonnet-4-20250514",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 15,
            "total_tokens": 27
        }
    }

    转换为 Anthropic 响应:
    {
        "id": "msg_xxx",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 12,
            "output_tokens": 15
        }
    }
    """
    choices = openai_response.get("choices", [])
    usage = openai_response.get("usage", {})

    # 提取文本内容
    text_content = ""
    finish_reason = None
    if choices:
        first_choice = choices[0]
        message = first_choice.get("message", {})
        text_content = message.get("content", "") or ""
        finish_reason = first_choice.get("finish_reason")

    # 生成 Anthropic 格式的 msg ID
    msg_id = _make_anthropic_id(openai_response.get("id", ""))

    # 使用原始请求中的模型名(Anthropic 格式)，而不是 OpenAI 映射后的名字
    model = original_model or openai_response.get("model", "")

    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text_content}],
        "model": model,
        "stop_reason": map_stop_reason(finish_reason),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def _make_anthropic_id(openai_id: str = "") -> str:
    """生成 Anthropic 风格的消息 ID: msg_xxxxx"""
    if openai_id and openai_id.startswith("msg_"):
        return openai_id
    # 使用 UUID 的一部分生成类似 Anthropic 的 ID
    short_id = uuid.uuid4().hex[:24]
    return f"msg_{short_id}"
