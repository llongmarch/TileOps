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

## Attention（待实现 · 热点）

尚未实现；子步骤暂用 `softmax`、`bmm` 等顶替。

- `scaled_dot_product_attention`
- `flash_attention` / `flash_attention_varlen`
- `paged_attention`
- `apply_rotary_pos_emb`
- `reshape_and_cache`
- `fp8_attention`（可选）

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

映射：`q = clamp(round(x / scale), -128, 127)`，`x̂ = float(q) * scale`（无 zero-point）。

### vLLM INT8（`tileops.quant_int8`）

| 算子 | 后端 |
|------|------|
| `input_to_int8` | PyTorch ⓡ |
| `per_token_quant_int8` | TileLang（per-row） |
| `per_token_group_quant_int8` | 基于 per-token |
| `block_dequant` | PyTorch ⓡ |
| `w8a8_block_int8_matmul` | PyTorch ⓡ |
| `apply_w8a8_block_int8_linear` | PyTorch ⓡ |

### vLLM FP8（`tileops.quant_fp8`）

| 算子 | 后端 |
|------|------|
| `is_fp8`, `default_fp8_dtype`, `get_fp8_min_max` | 工具函数 |
| `input_to_float8` | PyTorch ⓡ |
| `per_token_group_quant_fp8` | PyTorch ⓡ |
| `block_dequant_fp8` | PyTorch ⓡ |
| `w8a8_block_fp8_matmul` | PyTorch ⓡ |
| `w8a8_triton_block_scaled_mm` | 同上（vLLM 别名） |
| `apply_w8a8_block_fp8_linear` | PyTorch ⓡ |

### vLLM AWQ（`tileops.quant_awq`）

| 算子 | 后端 |
|------|------|
| `unpack_awq_int4`, `pack_awq_int4` | PyTorch ⓡ |
| `awq_dequantize` / `awq_dequantize_triton` | PyTorch ⓡ |
| `awq_gemm` / `awq_gemm_triton` | PyTorch ⓡ |

常量：`AWQ_TRITON_SUPPORTED_GROUP_SIZES`, `REVERSE_AWQ_ORDER`。

---

## Runtime（编译与调用，非算子）

`compile_prim`, `bench_ms`, `default_tilelang_target`, `default_torch_device`, `default_execution_backend`, `invoke_*`, `suggest_tile_config`, `torch_to_tl_dtype`, `target_kind`, `setup_metal_workarounds`, 等。

---

## 模块与后端对应

| 用户 API | TileLang 后端（如有） |
|----------|----------------------|
| `tileops.binary` | `tileops.cuda.binary`, `tileops.metal.binary` |
| `tileops.activation` | `tileops.cuda.activation`, `tileops.metal.activation` |
| `tileops.blas` | `tileops.cuda.blas`, `tileops.metal.blas` |
| `tileops.reduction` | `tileops.cuda.reduce`, `tileops.metal.reduce` |
| `tileops.norm` | `tileops.cuda.norm`, `tileops.metal.norm` |
| `tileops.fused` | `tileops.cuda.fused`, `tileops.metal.fused` |
| `tileops.quant` | `tileops.cuda.quant`, `tileops.metal.quant` |
| `tileops.conv` | `tileops.cuda.conv`, `tileops.metal.conv` |

---

## 规划中（其他 · 尚未实现）

- Pointwise：`exp`, `log`, `rsqrt`, `clamp`, 位运算等
- 张量操作：`gather`, `scatter`, `where`, `masked_fill`, …
- `group_norm`, `cross_entropy_loss`
- 更多 attention 见上节
