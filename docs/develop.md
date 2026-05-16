# 开发计划 — 待实现算子（纯推理场景）

本文档从**纯推理**角度列出深度学习常见但尚未在 TileOps 中实现的算子。

> **推理前提**：不考虑训练相关算子（损失函数、反向传播），只关注前向推理路径。

通过 `from tileops import …` 使用已实现的算子；完整列表见 `tileops.__all__` 和 [`docs/ops.md`](./ops.md)。

---

## 优先级说明

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

### 三角函数

| 算子 | 用途 |
|------|------|
| `sin` | 正弦（RoPE 位置编码） |
| `cos` | 余弦（RoPE 位置编码） |

### 归一化（Normalization — 推理路径）

| 算子 | 用途 |
|------|------|
| `batch_norm` | Batch Normalization（eval 模式，使用 running mean/var） |
| `group_norm` | Group Normalization（ViT、Detectron2、Diffusion 模型） |

### Padding / Pooling

| 算子 | 用途 |
|------|------|
| `pad` | 边界填充（`constant`、`reflect`、`replicate` 模式，卷积前填充） |
| `max_pool2d` / `max_pool1d` | 最大池化（卷积网络推理） |
| `avg_pool2d` / `avg_pool1d` | 平均池化（卷积网络推理） |

---

## 🟡 P1 — 中优先级

### 池化与采样

| 算子 | 用途 |
|------|------|
| `adaptive_avg_pool2d` | 自适应平均池化（分类头、特征图下采样） |

### 数值检测

| 算子 | 用途 |
|------|------|
| `isnan` / `isinf` / `isfinite` | 数值稳定性检测（推理 Debug） |

---

## 🟢 P2 — 低优先级

### 张量操作

| 算子 | 用途 |
|------|------|
| `upsample` / `interpolate` | 上采样（双线性/最近邻，分割/生成模型） |
| `masked_select` | 条件选择（flattened output） |
| `one_hot` | 独热编码 |

### 归一化

| 算子 | 用途 |
|------|------|
| `instance_norm` | Instance Normalization（风格迁移，推理场景较少） |

### 位运算

| 算子 | 用途 |
|------|------|
| `bitwise_and` / `or` / `xor` | int 类型位运算 |
| `bitwise_not` | 按位取反 |

> **注**：`logical_and` / `logical_or` / `logical_xor` 已实现于 `tileops.binary`，但输出为 float32（0.0/1.0），非整数位运算。

---

## 不在推理范围内的算子（不纳入开发计划）

以下算子仅为训练/反向传播设计，纯推理场景无需实现：

- 所有损失函数：`cross_entropy_loss`、`nll_loss`、`mse_loss`、`l1_loss`、`kl_div`、`binary_cross_entropy`、`hinge_loss`、`smooth_l1_loss`
- 反向传播相关：`scatter_add`（主要用于 gradient scatter）

---

## 已实现算子速览

以下算子无需重复实现：

### BLAS（8）

`gemm`, `gemv`, `mm`, `mv`, `outer`, `bmm`, `addmm`, `baddbmm`

### Convolution（2）

`conv1d`, `conv2d`

### Indexing（3）

`gather`, `index_select`, `nonzero`

### Condition（2）

`where`, `masked_fill`

### Matrix（2）

`tril`, `triu`

### Shape（9）

`cat`, `stack`, `split`, `chunk`, `permute`, `transpose`, `flip`, `repeat`, `expand`

### Sort（2）

`sort`, `argsort`

### Activation（18）

`relu`, `sigmoid`, `tanh`, `gelu`, `gelu_exact`, `silu`, `hardswish`, `hardsigmoid`, `leaky_relu`, `relu6`, `elu`, `selu`, `celu`, `hardtanh`, `softplus`, `mish`, `softsign`, `log_sigmoid`

### Unary — 数学（13）

`exp`, `log`, `sqrt`, `rsqrt`, `square`, `abs`, `sign`, `neg`, `round`, `floor`, `ceil`, `reciprocal`, `clamp`

### Binary（22）

`add`, `sub`, `mul`, `div`, `pow`, `fmod`, `remainder`, `floor_div`, `maximum`, `minimum`, `eq`, `ne`, `gt`, `ge`, `lt`, `le`, `atan2`, `copysign`, `hypot`, `xlogy`, `logical_and`, `logical_or`, `logical_xor`

### Reduction（10）

`reduce_sum`, `reduce_mean`, `reduce_prod`, `reduce_amax`, `reduce_amin`, `reduce_argmax`, `reduce_argmin`, `reduce_all`, `reduce_any`, `reduce_cumsum`

### Top-k（1）

`topk`

### Norm / Softmax（8）

`softmax`, `safe_softmax`, `online_softmax`, `log_softmax`, `layer_norm`, `rms_norm`, `skip_layer_norm`, `skip_rms_norm`

### Fused（2）

`silu_and_mul`, `gelu_and_mul`

### Quant（约 25）

`symm INT8 (4)`, `vLLM INT8 (6)`, `vLLM FP8 (9)`, `vLLM AWQ (6)`

---

## 后端模块对照

| 用户 API | TileLang 后端 |
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
