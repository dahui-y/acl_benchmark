"""HF 端点选择：两个端点都是间歇性的，单次探测不算证据。

三轮实测，同一台机器：

    第一轮  net_probe.sh   mirror HEAD 200      直连 HEAD 200
    第二轮  eval_assets    mirror 握手超时      直连 取到 64 字节
    第三轮  eval_assets    mirror 取到 64 字节  直连 SSL UNEXPECTED_EOF

我据此先后写下过"mirror 不通、直连优先"和它的反面，两次都是**拿一个样本
下全称结论**。正确的读法是：**两个端点都间歇可用，没有哪个更好。**
所以不选端点，改成每个端点重试若干次，谁先成谁上；跑长任务时中途掉了
还要能换端点重来。

（这和更早那次一模一样：一次 CLIP 拉取失败 -> "服务器拿不到评测数据"。
  同一个错误犯第三遍了，所以把它固化成代码，不再靠记性。）

    from hfnet import pick_endpoint
    ep = pick_endpoint()          # 设好 HF_ENDPOINT / HF_HUB_OFFLINE，返回端点
"""

import os
import urllib.request

DIRECT = "https://huggingface.co"
MIRROR = "https://hf-mirror.com"
ENDPOINTS = [DIRECT, MIRROR]

# 一个确定存在的小文件，用来验活
_PROBE = "/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/model_index.json"


def _alive(base, timeout=20):
    try:
        req = urllib.request.Request(base + _PROBE, headers={"Range": "bytes=0-63"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return len(r.read(64)) > 0
    except Exception:
        return False


def pick_endpoint(tries=3, verbose=True):
    """轮流试两个端点，共 tries 轮。返回第一个活的；都不活抛异常。"""
    os.environ["HF_HUB_OFFLINE"] = "0"      # env.sh 里是 1，必须硬覆盖
    for t in range(tries):
        for base in ENDPOINTS:
            if _alive(base):
                os.environ["HF_ENDPOINT"] = base
                if verbose:
                    print(f"[hfnet] 第 {t+1} 轮命中 {base}")
                return base
        if verbose:
            print(f"[hfnet] 第 {t+1} 轮两个端点都没通，重试")
    raise RuntimeError(
        f"{tries} 轮内 {ENDPOINTS} 都不可用。这次是真的不通（不是端点选错），"
        "隔几分钟再跑；间歇性掉线本身是已知情况。")
