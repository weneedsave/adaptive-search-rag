#!/usr/bin/env bash
# 评测专用入口。硬编码 .venv-eval 解释器，避免手滑在 .venv 里装评测依赖。
set -euo pipefail
cd "$(dirname "$0")/.."        # 回项目根：chroma_db / data 都是相对路径
export HF_HUB_OFFLINE=1        # HF 被墙，模型已缓存
export HF_HUB_DISABLE_XET=1    # 否则走 cas-server.xethub.hf.co 报 401
exec ./.venv-eval/Scripts/python.exe -m evaluation.runner "$@"
