"""
SSE 流式转换 - 将 OpenAI 流式响应转换为 Anthropic SSE 格式

OpenAI 流式格式:
  data: {"choices":[{"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n
  data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}\n\n
  data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n
  data: [DONE]\n\n

Anthropic 流式格式:
  event: message_start
  data: {"type":"message_start","message":{...}}\n\n
  event: content_block_start
  data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n
  event: ping
  data: {"type":"ping"}\n\n
  event: content_block_delta
  data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n\n
  event: content_block_stop
  data: {"type":"content_block_stop","index":0}\n\n
  event: message_delta
  data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":15}}\n\n
  event: message_stop
  data: {"type":"message_stop"}\n\n
"""

import json
import time
from typing import AsyncIterator

from converter import map_stop_reason, _make_anthropic_id


def _sse_event(event_type: str, data: dict) -> str:
    """格式化一个 Anthropic SSE 事件"""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def convert_stream(
    openai_stream: AsyncIterator[bytes],
    original_model: str = "",
) -> AsyncIterator[str]:
    """
    将 OpenAI SSE 流逐 chunk 转换为 Anthropic SSE 格式。

    Args:
        openai_stream: OpenAI SSE 响应的异步字节流
        original_model: 原始 Anthropic 请求中的模型名

    Yields:
        Anthropic SSE 格式的字符串事件
    """
    msg_id = _make_anthropic_id()
    model = original_model
    is_first_chunk = True
    input_tokens = 0
    output_tokens = 0
    buffer = ""  # 用于处理跨 chunk 的不完整行

    async for raw_bytes in openai_stream:
        # 将字节解码并与上次残留的 buffer 拼接
        text = buffer + raw_bytes.decode("utf-8", errors="replace")
        buffer = ""

        # 按行分割处理
        lines = text.split("\n")

        # 如果最后一行不以换行结束，保留为 buffer
        if not text.endswith("\n"):
            buffer = lines.pop()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 处理 data: [DONE] — 终止信号
            if line == "data: [DONE]":
                # 发送 content_block_stop + message_delta + message_stop
                yield _sse_event("content_block_stop", {
                    "type": "content_block_stop",
                    "index": 0,
                })
                yield _sse_event("message_delta", {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": "end_turn",
                        "stop_sequence": None,
                    },
                    "usage": {
                        "output_tokens": output_tokens,
                    },
                })
                yield _sse_event("message_stop", {
                    "type": "message_stop",
                })
                return

            # 只处理 data: 开头的行
            if not line.startswith("data: "):
                continue

            json_str = line[6:]  # 去掉 "data: " 前缀

            try:
                chunk = json.loads(json_str)
            except json.JSONDecodeError:
                continue

            # 提取流 chunk 中的 ID 和模型
            chunk_id = chunk.get("id", "")
            chunk_model = chunk.get("model", model)
            if not model:
                model = chunk_model

            choices = chunk.get("choices", [])

            # 处理 usage-only chunk (OpenAI stream_options.include_usage)
            if not choices and "usage" in chunk:
                usage = chunk["usage"]
                input_tokens = usage.get("prompt_tokens", input_tokens)
                output_tokens = usage.get("completion_tokens", output_tokens)
                continue

            if not choices:
                continue

            first_choice = choices[0]
            delta = first_choice.get("delta", {})
            finish_reason = first_choice.get("finish_reason")

            # ---- 首个 chunk: 发送 message_start + content_block_start + ping ----
            if is_first_chunk:
                is_first_chunk = False

                # message_start
                yield _sse_event("message_start", {
                    "type": "message_start",
                    "message": {
                        "id": msg_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": original_model or model,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {
                            "input_tokens": input_tokens,
                            "output_tokens": 0,
                        },
                    },
                })

                # content_block_start
                yield _sse_event("content_block_start", {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "text",
                        "text": "",
                    },
                })

                # ping
                yield _sse_event("ping", {"type": "ping"})

            # ---- 内容 delta ----
            content = delta.get("content")
            if content:
                output_tokens += 1  # 粗略估算，每个 delta 大致对应 1 个 token
                yield _sse_event("content_block_delta", {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "text_delta",
                        "text": content,
                    },
                })

            # ---- 结束信号 ----
            if finish_reason is not None:
                stop_reason = map_stop_reason(finish_reason)

                yield _sse_event("content_block_stop", {
                    "type": "content_block_stop",
                    "index": 0,
                })
                yield _sse_event("message_delta", {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": stop_reason,
                        "stop_sequence": None,
                    },
                    "usage": {
                        "output_tokens": output_tokens,
                    },
                })
                yield _sse_event("message_stop", {
                    "type": "message_stop",
                })
                return
