## BLAS类算子（基础线性代数）

- mm / addmm / bmm：基础矩阵乘法/带偏置
- mv / outer：矩阵-向量乘/外积


## Pointwise类算子（逐元素运算）

- 基础运算：add, sub, mul, div, pow
- 激活函数：relu, gelu, silu, sigmoid, tanh
- 数学运算：exp, log, rsqrt, reciprocal, abs, neg
- 逻辑运算：gt, lt, ge, le, eq, ne, logical_and, logical_or, logical_not
- 其他：clamp, triu, bitwise operations


## Reduction类算子（规约运算）

- mean / sum / prod
- min / max / amax
- argmin / argmax
- any / all
- cumsum


## 张量操作

- where / masked_fill / masked_select
- arange / ones / zeros
- index_select / gather / scatter
- tile / repeat / reshape


## 融合算子与高阶运算

- layernorm / rms_norm / group_norm
- softmax / log_softmax
- cross_entropy_loss
- fused: gelu_and_mul, silu_and_mul
- fused: skip_rms_norm, skip_layer_norm


## 特殊算子

- apply_rotary_position_embedding (RoPE)
- unique (Unique elements)
- Sparse Attention (as in DeepSeek-V4)
- Hadamard Transform (as in DeepSeek-V4)
- FP8 MatMul (as in DeepSeek-V4)
