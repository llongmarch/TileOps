# TileOps 算子文档

通过 `from tileops import …` 使用；完整列表见 `tileops.__all__`。

**后端**：多数算子由 TileLang 编译至 `tileops.cuda.*` / `tileops.metal.*`；部分委托给 PyTorch（标注 ⓡ）。

---

# 已实现算子

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

委托给 PyTorch ⓡ：

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

## Pad（边界填充）

| 算子 | 说明 | 后端 |
|------|------|------|
| `pad` | 边界填充（`constant` / `reflect` / `replicate` / `circular`） | TileLang (`cuda.pad`, `metal.pad`) + PyTorch ⓡ fallback |

`constant` 模式且填充最后 1~2 维时使用 TileLang kernel；其他模式和更高维 padding 自动回退到 PyTorch。

---

## Interpolate（插值 / 缩放）

委托给 PyTorch ⓡ：

| 算子 | 说明 |
|------|------|
| `interpolate` | 张量缩放（`nearest` / `bilinear` / `bicubic` / `trilinear` / `area` 等） |

支持 `size` 或 `scale_factor` 指定输出尺寸，以及 `align_corners`、`antialias` 参数。

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
| 指数/对数 | `exp`, `log`, `exp2`, `exp10`, `log2`, `log10`, `log1p` |
| 三角 | `sin`, `cos`, `tan` |
| 反三角 | `asin`, `acos`, `atan` |
| 双曲 | `sinh`, `cosh` |
| 反双曲 | `asinh`, `acosh`, `atanh` |
| 幂/根 | `sqrt`, `rsqrt`, `square` |
| 误差函数 | `erf` |
| 符号/绝对值 | `abs`, `sign`, `neg` |
| 特殊值检测 | `isnan`, `isinf`, `isfinite` |
| 取整 | `round`, `floor`, `ceil`, `trunc` |
| 倒数 | `reciprocal` |
| 位运算 | `bitwise_not` |
| 裁剪 | `clamp(min, max)` |

---

## Binary（逐元素二元）

**算术**：`add`, `sub`, `mul`, `div`, `pow`, `fmod`, `remainder`, `floor_div`

**比较**：`eq`, `ne`, `gt`, `ge`, `lt`, `le`

**极值**：`maximum`, `minimum`

**数学**：`atan2`, `copysign`, `hypot`, `xlogy`, `logaddexp`

**逻辑**：`logical_and`, `logical_or`, `logical_xor`

**位运算**：`bitwise_and`, `bitwise_or`, `bitwise_xor`, `shift_left`, `shift_right`

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

## Softmax

| 算子 | 说明 |
|------|------|
| `softmax` | Softmax（两遍稳定） |
| `safe_softmax` | 与 `softmax` 同语义的显式别名 |
| `online_softmax` | 单遍在线 Softmax |
| `log_softmax` | Log-Softmax |

---

## Norm（归一化）

| 算子 | 说明 |
|------|------|
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

# 待实现算子

纯推理场景，按优先级排列。

| 标记 | 含义 |
|:----:|------|
| 🔴 P0 | LLM / 视觉推理中高频依赖 |
| 🟡 P1 | 通用模型推理中常用 |
| 🟢 P2 | 特定场景需求 |

---

## 🔴 P0 — 高优先级

### Attention（LLM 推理核心）

| 算子 | 用途 |
|------|------|
| `scaled_dot_product_attention` | 融合 SDPA（PyTorch 2.0 兼容） |
| `flash_attention` / `flash_attention_varlen` | 变长 Flash Attention |
| `paged_attention` | vLLM 分页注意力 |
| `apply_rotary_pos_emb` | RoPE 旋转位置编码 |
| `reshape_and_cache` | KV cache 管理 |

### 归一化

| 算子 | 用途 |
|------|------|
| `batch_norm` | Batch Normalization（eval 模式） |
| `group_norm` | Group Normalization（ViT / Diffusion） |

### Pooling

| 算子 | 用途 |
|------|------|
| `max_pool2d` / `max_pool1d` | 最大池化 |
| `avg_pool2d` / `avg_pool1d` | 平均池化 |

---

## 🟡 P1 — 中优先级

| 算子 | 用途 |
|------|------|
| `adaptive_avg_pool2d` | 自适应平均池化（分类头） |

---

## 🟢 P2 — 低优先级

| 算子 | 用途 |
|------|------|
| `masked_select` | 条件选择（flattened output） |
| `one_hot` | 独热编码 |
| `instance_norm` | Instance Normalization（风格迁移） |

---

## 不在推理范围内的算子

以下仅为训练/反向传播设计，纯推理场景无需实现：

- 损失函数：`cross_entropy_loss`、`nll_loss`、`mse_loss`、`l1_loss`、`kl_div`、`binary_cross_entropy`、`hinge_loss`、`smooth_l1_loss`
- 反向传播：`scatter_add`（主要用于 gradient scatter）

---

# 模块与后端对应

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
| `tileops.pad` | `tileops.cuda.pad`, `tileops.metal.pad` + PyTorch ⓡ |
| `tileops.interpolate` | PyTorch ⓡ |
| `tileops.softmax` | `tileops.cuda.softmax`, `tileops.metal.softmax` |
| `tileops.norm` | `tileops.cuda.norm`, `tileops.metal.norm` |
| `tileops.fused` | `tileops.cuda.fused`, `tileops.metal.fused` |
| `tileops.quant` | `tileops.cuda.quant`, `tileops.metal.quant` |
