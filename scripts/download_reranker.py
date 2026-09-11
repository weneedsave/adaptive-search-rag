"""一次性下载 bge-reranker-v2-m3 到本地 HF 缓存。

HF 直连被墙，走 hf-mirror 镜像。下好之后 CrossEncoder 就能用
local_files_only=True 离线加载，不再依赖网络。

用法： .venv/Scripts/python.exe scripts/download_reranker.py
"""
import os
import sys
import time

# 必须在 import huggingface_hub 之前设置，否则镜像不生效
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 关键：hub 1.x 默认走 Xet 存储后端（cas-server.xethub.hf.co），
# hf-mirror 不镜像它 → 401 Unauthorized。关掉，强制走传统 HTTP 通道。
os.environ["HF_HUB_DISABLE_XET"] = "1"

sys.stdout.reconfigure(encoding="utf-8")

from huggingface_hub import snapshot_download  # noqa: E402

MODEL_NAME = "BAAI/bge-reranker-v2-m3"


def main() -> None:
    print(f"从镜像 {os.environ['HF_ENDPOINT']} 下载 {MODEL_NAME} ...")
    t0 = time.time()
    path = snapshot_download(
        repo_id=MODEL_NAME,
        # 跳过用不到的大文件，省流量：tf/onnx/flax 权重都不要
        ignore_patterns=["*.h5", "*.msgpack", "*.onnx", "onnx/*", "tf_model.h5"],
        max_workers=4,
    )
    print(f"完成，耗时 {time.time() - t0:.1f}s")
    print(f"本地路径：{path}")


if __name__ == "__main__":
    main()
