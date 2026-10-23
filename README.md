# 4096 个 FP16 的 RMSNorm，在这台 Mac 上怎么做最快

结论先说（N = 4096，一行 8 KiB）：

- **单次调用并等结果**：手写 NEON 内核 **0.73 µs**（无 weight）/ **0.93 µs**（有 weight）。同一条件下 PyTorch MPS 约 **202 µs**，因为一次 Metal 提交+同步就要这么久。**单行无脑选 CPU。**
- **很多行**：单次同步下，≤1024 行 NEON 仍然最快；4096 行有 weight 时 NEON、MPS、MLX 基本打平（~0.9 ms）；16384 行 MPS 最快（2.87 ms vs NEON 3.29 ms）。
- **如果是流水线连续提交**（不每行同步）：MPS 的摊薄成本可以低到单行 **2.9 µs**、4096 行 **575 µs**；但这是吞吐，不是一次调用的延迟。
- **PyTorch MPS 的坑**：`F.rms_norm(x, ..., weight=None)` 在 MPS 上会走很慢的 composite 路径（4096 行要 6.7 ms）。即使没有 weight，也传一个全 1 的 weight，才会走 fused kernel。

测试机器：

| 项目 | 值 |
|---|---|
| SoC | Apple M5，10 核 CPU（4P+6E），10 核 GPU |
| 内存 | 16 GB |
| OS | macOS 26.6.2 |
| Python | 3.13.15 |
| PyTorch | 2.13.0（MPS） |
| MLX | 0.32.4 |
| 编译器 | Apple clang 21 |

## 单次调用延迟（每次调用后同步）

单位 µs/次，包含 Python/框架调用开销；custom 使用预分配输出。这是“算完这一次、马上要结果”的延迟。

### 无 weight

| rows | custom NEON | torch CPU | torch MPS | MLX GPU | 最快 |
|---:|---:|---:|---:|---:|---|
| 1 | **0.73** | 4.87 | 214.11 | 203.10 | custom |
| 8 | **2.85** | 31.63 | 227.93 | 204.26 | custom |
| 64 | **13.45** | 90.59 | 291.69 | 208.92 | custom |
| 256 | **40.50** | 203.72 | 510.89 | 237.16 | custom |
| 1024 | **110.75** | 1343.48 | 1785.98 | 368.60 | custom |
| 4096 | **735.85** | 5974.30 | 6741.82 | 992.45 | custom |
| 16384 | **2998.03** | 26484.81 | 27110.71 | 3070.03 | custom |

### 有 weight

| rows | custom NEON | torch CPU | torch MPS | MLX GPU | 最快 |
|---:|---:|---:|---:|---:|---|
| 1 | **0.93** | 6.07 | 201.86 | 200.48 | custom |
| 8 | **3.86** | 35.95 | 202.01 | 199.78 | custom |
| 64 | **20.44** | 112.65 | 217.78 | 209.94 | custom |
| 256 | **50.23** | 267.35 | 238.17 | 228.74 | custom |
| 1024 | **160.59** | 1773.39 | 405.79 | 395.03 | custom |
| 4096 | 908.50 | 7743.48 | 904.77 | **898.71** | MLX（三者基本打平） |
| 16384 | 3290.58 | 28605.74 | **2870.69** | 2931.43 | MPS |

> GPU 每次单独提交的延迟约 200 µs，基本是固定开销；小 batch 下这个开销占绝对主导。

## 流水线吞吐（连续提交，最后同步一次）

如果你是在一个模型里连续跑很多 kernel，GPU 的提交可以重叠，数字会好看很多：

| rows | custom NEON (无 weight) | torch MPS (无 weight) | custom NEON (有 weight) | torch MPS (有 weight) |
|---:|---:|---:|---:|---:|
| 1 | **0.71** | 13.26 | **0.87** | 2.91 |
| 8 | **2.82** | 20.79 | 3.78 | **2.83** |
| 64 | **13.71** | 63.58 | 17.92 | **5.80** |
| 1024 | **110.63** | 1329.59 | 151.44 | **110.55** |
| 4096 | **575.27** | 5449.34 | 638.50 | **574.81** |
| 16384 | 2468.24 | 22699.13 | 2725.39 | **2536.54** |

所以：

- **独立一次 RMSNorm**：CPU NEON。
- **有 weight、几十行以上的 GPU 流水线**：MPS fused 很快（8 行 2.8 µs/行，4096 行 575 µs 总计）。
- **无 weight 的 MPS**：除非显式传全 1 weight，否则不要用。

## 为什么这样最快

1. **单行只有 8 KiB，瓶颈是 launch/dispatch，不是算力。**
   GPU 一次 command buffer 提交+同步要几百微秒，4096 个元素根本不值得。CPU 上数据在 L1/L2 里，直接算最快。

2. **FP16 用 FMLAL 做 widening FMA，而不是先转 FP32 再乘。**
   Apple Silicon 支持 `FP16FML`：
   ```c
   a0 = vfmlalq_low_f16(a0, x0, x0);
   a0 = vfmlalq_high_f16(a0, x0, x0);
   ```
   一条指令吃掉 4 个 half 的平方并累加到 FP32。平方和在 FP32 中累加，避免 FP16 溢出和精度损失。

3. **64 个元素一组、8 个累加器。**
   足够长的依赖链被拆开，FMA 流水线不会等前一次累加；4096 正好是 64 的整数倍，没有尾部。

4. **两遍走：先求和，再缩放。**
   8 KiB 输入留在 L1，第二遍读几乎免费；这比为了单遍而缓存 4096 个 FP32 更划算。

5. **batch 用 `dispatch_apply` 按行切分。**
   每行独立，天然并行；4 个性能核在 1024 行时能到 ~150 GB/s。行数少于 64 时不启线程，避免 dispatch 开销。

## 精度

用 FP32 参考实现对比，随机输入、4096 个元素：

- 无 weight：max abs error ≈ **0.0010**
- 有 weight：max abs error ≈ **0.0019**

误差来自最终写回 FP16，以及 FP32 累加顺序不同；不是用 FP16 做平方和。

## 怎么复现

```bash
make
.venv/bin/python benchmark.py --weight                 # 单次延迟
.venv/bin/python benchmark.py --weight --queued        # 流水线吞吐
```

如果还没有 MLX 环境：

```bash
python3 -m venv .venv --system-site-packages
.venv/bin/pip install mlx
```

只跑 C 内核（不含 Python 开销）：

```bash
make bench
# 或指定 rows / iters / threads / weight
./build/bench_rmsnorm 1 300000 1 0
./build/bench_rmsnorm 4096 1000 4 1
```

## 文件

| 文件 | 作用 |
|---|---|
| `csrc/rmsnorm.c` | NEON FP16 RMSNorm：单行 + 多线程 batch |
| `csrc/bench.c` | 纯 C 微基准，测内核本身 |
| `benchmark.py` | 与 PyTorch CPU/MPS、MLX 对比；`--queued` 切换吞吐模式 |
| `results_latency.json` | 单次调用延迟原始结果 |
| `results_weight.json` | 流水线吞吐原始结果 |
| `benchmark_latency.out` / `benchmark_weight.out` | 本次完整输出 |

## 怎么在代码里用

```python
import ctypes, torch

lib = ctypes.CDLL("build/librmsnorm.dylib")
lib.rmsnorm_f16_batch.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int,
]

x = torch.randn(1, 4096, dtype=torch.float16).contiguous()
w = torch.randn(4096, dtype=torch.float16).contiguous()
y = torch.empty_like(x)

lib.rmsnorm_f16_batch(
    x.data_ptr(), w.data_ptr(), y.data_ptr(),
    1, 4096, 1e-5, 1,
)
```

要求：CPU 上的连续 FP16 张量；`y` 预分配。内核也支持任意 `n`（尾部有标量兜底），但性能是按 `n = 4096` 调的。

## 什么时候该用 GPU

- 单行 / 几行：**CPU NEON**，别上 GPU。
- 有 weight、几十行以上、并且能连续提交：**MPS fused RMSNorm**，吞吐最好。
- 无 weight、几千行：**NEON 多线程**，因为 PyTorch MPS 的无 weight 路径目前很慢。
- 上万行、且单次同步：MPS 略胜；这时选框架自带的 fused 实现更省事。

这些数字是“本机 + 当前 PyTorch/MLX 版本”的结果；MPS 的 kernel 选择和版本关系很大，升级后建议重跑 `benchmark.py`。
