"""
配置管理 - 通过环境变量或 .env 文件配置代理服务
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
#  Provider 配置（多后端）
# ============================================================
#
# 注意：
# - 这里预置的是“公开可查的默认 base url + 标准路径”与占位 model/key。
# - 你下次只需要用环境变量把对应的 *_API_KEY / *_MODEL 覆盖即可。
#
# OpenAI 兼容接口期望：POST {BASE_URL}/v1/chat/completions
#

# ===== Kimi Code（OpenAI 兼容，占位）=====
# 文档: https://www.kimi.com/code/docs/more/third-party-agents.html
# OpenAI 兼容 Base URL: https://api.kimi.com/coding/v1
# 模型: kimi-for-coding
KIMI_CODE_BASE_URL = os.getenv("KIMI_CODE_BASE_URL", "https://api.kimi.com/coding/v1")
KIMI_CODE_API_KEY = os.getenv("KIMI_CODE_API_KEY", "")
KIMI_CODE_MODEL = os.getenv("KIMI_CODE_MODEL", "kimi-for-coding")

# ===== Moonshot（通用 OpenAI 兼容，占位）=====
MOONSHOT_BASE_URL = os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.cn")
MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY", "")
# 具体可用模型名以 Moonshot 控制台为准
MOONSHOT_MODEL = os.getenv("MOONSHOT_MODEL", "kimi-k2.5")

# ===== MiniMax（占位）=====
MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL", "https://api.minimax.chat")
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")
MINIMAX_MODEL = os.getenv("MINIMAX_MODEL", "")

# ===== GLM / Zhipu（占位）=====
ZHIPU_BASE_URL = os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn")
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")
ZHIPU_MODEL = os.getenv("ZHIPU_MODEL", "")


# ============================================================
#  默认后端配置（OpenAI 兼容）
# ============================================================

# 后端 API 完整地址 (直接到 chat/completions 端点)
OPENAI_API_URL = os.getenv(
    "OPENAI_API_URL",
    "",
)

# 后端 API Key
# 建议通过环境变量提供，避免写死在代码里
OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY",
    "",
)

# 后端使用的模型名
BACKEND_MODEL = os.getenv(
    "BACKEND_MODEL",
    "",
)

# ============================================================
#  代理服务配置
# ============================================================

SERVER_PORT = int(os.getenv("SERVER_PORT", "5588"))
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")

# 默认最大 token 数 (Anthropic API 中 max_tokens 是必填字段)
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "4096"))

# ============================================================
#  模型名称映射
# ============================================================
# Xcode 发来的 Anthropic 模型名 → 后端实际使用的模型名
# 所有 Anthropic 模型名都映射到同一个后端模型

_BACKEND_MODEL = BACKEND_MODEL

_default_model_mapping = {
    "claude-sonnet-4-20250514": _BACKEND_MODEL,
    "claude-opus-4-20250514": _BACKEND_MODEL,
    "claude-haiku-4-5-20251001": _BACKEND_MODEL,
    "claude-3-5-sonnet-20241022": _BACKEND_MODEL,
    "claude-3-5-haiku-20241022": _BACKEND_MODEL,
    "claude-3-opus-20240229": _BACKEND_MODEL,
    "claude-3-sonnet-20240229": _BACKEND_MODEL,
    "claude-3-haiku-20240307": _BACKEND_MODEL,
}


def get_model_mapping() -> dict:
    """获取模型名称映射表"""
    env_mapping = os.getenv("MODEL_MAPPING", "")
    if env_mapping:
        mapping = {}
        for pair in env_mapping.split(","):
            pair = pair.strip()
            if ":" in pair:
                src, dst = pair.split(":", 1)
                mapping[src.strip()] = dst.strip()
        return mapping
    return _default_model_mapping.copy()


def map_model_name(anthropic_model: str) -> str:
    """将 Anthropic 模型名映射为后端使用的模型名"""
    mapping = get_model_mapping()
    return mapping.get(anthropic_model, _BACKEND_MODEL)
