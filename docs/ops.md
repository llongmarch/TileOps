# TileOps 已支持算子

通过 `from tileops import …` 使用；完整列表见 `tileops.__all__`。

**后端**：多数算子由 TileLang 编译至 `tileops.cuda.*` / `tileops.metal.*`；部分量化路径为 PyTorch 参考实现（标注 ⓡ）。

---

## BLAS（线性代数）

| 算子 | 说明 |
|------|------|
| `gemm` | 通用 GEMM（底层） |
| `gemv` | 矩阵-向量乘（底层） |
| `mm` | `C = A @ B` |
| `mv` | `y = A @ x` |
| `outer` | 外积 |
| `bmm` | 批量矩阵乘 |
| `addmm` | `beta * C + alpha * (A @ B)` |
| `baddbmm` | 批量 `addmm` |

---

## Convolution（卷积）

| 算子 | 说明 |
|------|------|
| `conv1d` | 1-D 卷积（batch, channel, length） |
| `conv2d` | 2-D 卷积（batch, channel, height, width） |

支持 groups 分组卷积；bias 可选（默认全零）。

---

## Indexing（索引）

| 算子 | 说明 | 后端 |
|------|------|------|
| `gather` | 按索引收集元素 | TileLang (`cuda.indexing`, `metal.indexing`) |
| `index_select` | 沿 dim 按索引选择 | PyTorch ⓡ |
| `nonzero` | 非零元素索引 | PyTorch ⓡ |

---

## Condition（条件选择）

| 算子 | 说明 | 后端 |
|------|------|------|
| `where(cond, x, y)` | 条件选择 | TileLang (`cuda.condition`, `metal.condition`) |
| `masked_fill(x, mask, value)` | 掩码填值 | TileLang (kernel) + PyTorch ⓡ (broadcast) |

---

## Matrix（三角矩阵）

| 算子 | 说明 | 后端 |
|------|------|------|
| `tril` | 下三角矩阵 | TileLang (`cuda.matrix`, `metal.matrix`) |
| `triu` | 上三角矩阵 | TileLang |

支持 `diagonal` 参数，支持 batched 输入。

---

## Shape（形状操作）

所有算子委托给 PyTorch ⓡ：

| 算子 | 说明 |
|------|------|
| `cat` | 沿维拼接 |
| `stack` | 沿新维堆叠 |
| `split` | 按大小拆分 |
| `chunk` | 按块数拆分 |
| `permute` | 维度重排 |
| `transpose` | 两维转置 |
| `flip` | 沿维翻转 |
| `repeat` | 复制 |
| `expand` | 广播 |

---

## Sort（排序）

| 算子 | 说明 | 后端 |
|------|------|------|
| `sort` | 沿维排序 | TileLang (`cuda.sort`, `metal.sort`) |
| `argsort` | 排序索引 | TileLang |

支持 `descending` 参数和任意 `dim`。

---

## Unary（数学一元）

| 类别 | 算子 |
|------|------|
| 指数/对数 | `exp`, `log` |
| 幂/根 | `sqrt`, `rsqrt`, `square` |
| 符号/绝对值 | `abs`, `sign`, `neg` |
| 取整 | `round`, `floor`, `ceil` |
| 倒数 | `reciprocal` |
| 裁剪 | `clamp(min, max)` |

---

## Binary（逐元素二元）

**算术**：`add`, `sub`, `mul`, `div`, `pow`, `fmod`, `remainder`, `floor_div`

**比较**：`eq`, `ne`, `gt`, `ge`, `lt`, `le`

**极值**：`maximum`, `minimum`

**数学**：`atan2`, `copysign`, `hypot`, `xlogy`

**逻辑**：`logical_and`, `logical_or`, `logical_xor`

---

## Activation（逐元素一元）

| 类别 | 算子 |
|------|------|
| 经典 | `relu`, `sigmoid`, `tanh` |
| GELU | `gelu`, `gelu_exact` |
| Swish 族 | `silu`, `hardswish`, `hardsigmoid` |
| ReLU 变体 | `leaky_relu`, `relu6`, `elu`, `selu`, `celu`, `hardtanh` |
| 平滑 | `softplus`, `mish`, `softsign` |
| 对数 | `log_sigmoid` |

---

## Reduction（沿维规约）

| 算子 | 说明 |
|------|------|
| `reduce_sum` | 求和 |
| `reduce_mean` | 均值 |
| `reduce_prod` | 乘积 |
| `reduce_amax` | 最大绝对值 |
| `reduce_amin` | 最小值 |
| `reduce_argmax` | 最大值索引（`int64`） |
| `reduce_argmin` | 最小值索引（`int64`） |
| `reduce_all` | 全真 |
| `reduce_any` | 存在真 |
| `reduce_cumsum` | 前缀和 |
| `topk` | Top-*k*（返回 values + indices） |

---

## Norm / Softmax

| 算子 | 说明 |
|------|------|
| `softmax` | Softmax（两遍稳定） |
| `safe_softmax` | 与 `softmax` 同语义的显式别名 |
| `online_softmax` | 单遍在线 Softmax |
| `log_softmax` | Log-Softmax |
| `layer_norm` | LayerNorm |
| `rms_norm` | RMSNorm |
| `skip_layer_norm` | 融合 `layer_norm(x + residual, …)` |
| `skip_rms_norm` | 融合 `rms_norm(x + residual, …)` |

---

## Fused（LLM 融合）

| 算子 | 说明 |
|------|------|
| `silu_and_mul` | `silu(a) * b`（SwiGLU 门控） |
| `gelu_and_mul` | `gelu(a) * b`（GeGLU 门控） |

---

## Quant（量化）

### 对称 INT8（`tileops.quant` → TileLang）

| 算子 | 说明 |
|------|------|
| `quantize_per_tensor` | 全张量标量 scale → `int8` |
| `dequantize_per_tensor` | 标量反量化 |
| `quantize_per_channel` | 按通道 scale → `int8` |
| `dequantize_per_channel` | 按通道反量化 |

### vLLM INT8 / FP8 / AWQ

参见 `tileops.quant_int8`、`tileops.quant_fp8`、`tileops.quant_awq`。

---

## 模块与后端对应

| 用户 API | TileLang 后端（如有） |
|----------|----------------------|
| `tileops.binary` | `tileops.cuda.binary`, `tileops.metal.binary` |
| `tileops.activation` | `tileops.cuda.activation`, `tileops.metal.activation` |
| `tileops.unary` | `tileops.cuda.unary`, `tileops.metal.unary` |
| `tileops.blas` | `tileops.cuda.blas`, `tileops.metal.blas` |
| `tileops.conv` | `tileops.cuda.conv`, `tileops.metal.conv` |
| `tileops.indexing` | `tileops.cuda.indexing` (gather), PyTorch ⓡ (index_select, nonzero) |
| `tileops.condition` | `tileops.cuda.condition`, `tileops.metal.condition` |
| `tileops.matrix` | `tileops.cuda.matrix`, `tileops.metal.matrix` |
| `tileops.sort` | `tileops.cuda.sort`, `tileops.metal.sort` |
| `tileops.reduction` | `tileops.cuda.reduce`, `tileops.metal.reduce` |
| `tileops.norm` | `tileops.cuda.norm`, `tileops.metal.norm` |
| `tileops.fused` | `tileops.cuda.fused`, `tileops.metal.fused` |
| `tileops.quant` | `tileops.cuda.quant`, `tileops.metal.quant` |
