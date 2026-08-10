"""第二轮探测：net_probe.sh 之后剩下的两个真问题。

net_probe.sh 的结果把网络这条阻塞彻底解除了（huggingface.co / github.com
直连都 200）。它剩下两个 FAIL，两个都不是网络：

    401  nateraw/inception-v3-fid/...       仓库不存在（我编的名字）
    401  laion2B-en-aesthetic/part-000...   文件名是我编的哈希

    —— HF 对"不存在或门控"的资源统一回 401（避免泄露仓库是否存在），
       所以 401 只说明"这个 URL 不对"，不说明"下不来"。证据：同一个
       laion 仓库的 API 返回 200。

但 HEAD 200 也不等于能下载（可能只是网关在应答）。而且真正的风险不在
带宽，在**可比性**：ScaleDiff / DemoFusion 报的 FID 是对着他们那 1000 张
LAION 真图算的；LAION-5B 原始数据集 2023-12 已下架，重发为 relaion。
拿不到同一批真图，我们的绝对 FID 就和他们发表的数字不在同一把尺子上，
"其余 baseline 引用发表值"这个省 GPU 的方案就有裂缝。

所以这个脚本只干三件事，都不下载大东西：

    ① 真的下几个字节（不是 HEAD），分别走 hf-mirror 和直连，验证通道是实的；
    ② 用 API 列出候选 caption 源的真实文件名（不再猜哈希）；
    ③ 列出候选真图参考集，报告哪个拿得到。

    python scalediff_probe/eval_assets.py
"""

import os
import sys
import urllib.request

os.environ.setdefault("HF_HUB_OFFLINE", "0")
MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
DIRECT = "https://huggingface.co"

# 一个确定存在的小文件：SDXL 的 model_index.json（我们缓存里就有这个仓库）
SMALL = "/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/model_index.json"

# 候选 caption 源。不猜文件名 —— 用 API 列。
CAPTION_REPOS = [
    ("laion/relaion2B-en-research-safe", "LAION 下架后的官方重发"),
    ("laion/laion2B-en-aesthetic", "原始美学子集（API 说存在）"),
    ("laion/laion-coco", "LAION 图 + 合成 caption"),
]

# 候选真图参考集（FID 的 reference）。
IMAGE_REPOS = [
    ("sayakpaul/coco-30-val-2014", "SD 系评测最常用的 COCO val 打包"),
    ("nlphuji/mscoco_2014_5k_test_image_text_retrieval", "COCO 5k 测试集"),
    ("laion/relaion2B-en-research-safe", "同上，若含图"),
]


def real_get(url, nbytes=64):
    """真的取前几个字节，不是 HEAD。"""
    try:
        req = urllib.request.Request(url, headers={"Range": f"bytes=0-{nbytes-1}"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return len(r.read(nbytes)), None
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}"


def main():
    print("\n== ① 通道是不是实的（真下字节，不是 HEAD）==")
    live = {}
    for name, base in (("hf-mirror", MIRROR), ("直连 huggingface.co", DIRECT)):
        n, err = real_get(base + SMALL)
        live[name] = n > 0
        print(f"  {'OK  ' if n else 'FAIL'} {name}: 取到 {n} 字节"
              + (f"   {err}" if err else ""))
    if not any(live.values()):
        print("\n  两条通道都下不来 —— net_probe 的 200 是网关在应答，"
              "结论要退回去。停在这里，不要继续规划。")
        return 1

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("\n  huggingface_hub 未装：pip install -U huggingface_hub "
              "-i https://pypi.tuna.tsinghua.edu.cn/simple")
        return 1
    api = HfApi(endpoint=MIRROR)

    def probe(repos, kind):
        got = []
        for rid, why in repos:
            try:
                info = api.repo_info(rid, repo_type="dataset", files_metadata=False)
                files = [s.rfilename for s in info.siblings or []]
                data = [f for f in files
                        if f.endswith((".parquet", ".tar", ".json", ".jsonl",
                                       ".csv", ".tsv", ".zip"))]
                print(f"  OK   {rid}\n       {why}   共 {len(files)} 个文件")
                for f in data[:4]:
                    print(f"       - {f}")
                if len(data) > 4:
                    print(f"       ... 还有 {len(data)-4} 个")
                got.append(rid)
            except Exception as e:
                msg = str(e).split("\n")[0][:110]
                print(f"  FAIL {rid}\n       {type(e).__name__}: {msg}")
        return got

    print(f"\n== ② caption 源（1000 条 prompt 从这里采）==")
    cap = probe(CAPTION_REPOS, "caption")

    print(f"\n== ③ 真图参考集（FID 的 reference）==")
    img = probe(IMAGE_REPOS, "image")

    print("\n== 判读 ==")
    print(f"  caption 源可用: {cap or '无'}")
    print(f"  真图参考集可用: {img or '无'}")
    print("""
  网络已经不是阻塞了。剩下的是**可比性**，那不是网络问题：
    - 拿得到同源 caption + 真图 -> 我们复现的 ScaleDiff 行应当接近其发表值，
      接近则管线通过校准，其余 baseline 可以引用发表数字；
    - 拿不到同源 -> 绝对 FID 不可比，只能两个 arm 都自己跑、报**相对**变化，
      并在表里明说参考集不同。这不致命（我们的主张是 A/B 差值），
      但必须事前写死，不能等数字出来再选口径。""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
