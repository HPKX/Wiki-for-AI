# MindSpeed TE 实现完整梳理（对照 TE 原生 MXFP8 方案）

---

## 一、整体架构对比

### TE 原生架构

```
用户代码 (autocast + recipe)
    ↓
Python 控制层
  autocast() → FP8GlobalStateManager → BasicLinear
                    ↓                       ↓
              MXFP8Quantizer          general_gemm()
    ↓
C++/CUDA 执行层
  tex.quantize() → quantize_mxfp8_kernel (CUDA)
  tex.generic_gemm() → cuBLAS MXFP8 GEMM
  tex.dequantize() → dequantize_mxfp8_kernel
```

### MindSpeed 架构

```
用户代码 (autocast + recipe)     ← 接口签名相同，namespace 被 patch 劫持
    ↓
Python 控制层 (全部重写)
  fp8_autocast() → FP8GlobalStateManager → TEColumnParallelLinear / TERowParallelLinear
                        ↓                           ↓
                  FP8Metadata.quantization()    fp8_matmul() / fp8_matmul_add()
                        ↓                           ↓
                  Recipe.quantization()        tensor.quant_matmul()
    ↓
torch_npu 算子层 (替代 C++/CUDA)
  npu_dynamic_mx_quant[_with_dual_axis]()  ← 替代 tex.quantize()
  npu_quant_matmul()                       ← 替代 tex.generic_gemm() / cuBLAS
```

**核心差异**：TE 是 Python → C++ binding → CUDA kernel 三层；MindSpeed 是 Python → torch_npu 算子两层，中间没有 C++ 绑定。

### Patch 机制

MindSpeed 不依赖 TE 的实际代码，但复用 TE 的 import 路径作为"挂载点"：

```python
# transformer_engine_basic.py
pm.register_patch('transformer_engine.pytorch.fp8_autocast', fp8_autocast)
pm.register_patch('transformer_engine.common.recipe.MXFP8BlockScaling', MXFP8BlockScaling)
pm.register_patch('transformer_engine.pytorch.tensor.QuantizedTensor', torch.nn.Module, create_dummy=True)
```

- `create_dummy=True`：TE 未安装时创建空壳模块，再挂载 MindSpeed 实现
- TE 已安装时：原有类被 MindSpeed 实现整体替换
- 运行时 `from transformer_engine.common.recipe import MXFP8BlockScaling` 解析到的是 MindSpeed 的类

| 依赖 | 是否必需 | 说明 |
|------|---------|------|
| Transformer Engine 代码 | **不需要** | `create_dummy=True` 可创建空壳模块 |
| Transformer Engine import 路径 | **需要**（作为挂载点） | 所有 patch 都挂在 `transformer_engine.*` 路径下 |
| Megatron-Core | **需要** | `megatron.core.*` 的调用链是真正的入口 |
| torch_npu | **需要** | 所有底层算子的实际提供者 |

---

## 二、核心组件逐一对照

### 2.1 autocast 入口

| | TE 原生 | MindSpeed |
|---|---------|-----------|
| **入口** | `quantization.py:563` `autocast()` | `fp8.py:12` `fp8_autocast()` |
| **进入时** | `FP8GlobalStateManager.autocast_enter()` 设置 `FP8_ENABLED`, `FP8_RECIPE`, 检查硬件 | 同名方法，设置相同字段 |
| **退出时** | `reduce_and_update_fp8_tensors()` — 对 MXFP8 是 NOOP | `fp8_autocast_exit()` — 只对 `DelayedScalingRecipe` 执行 `finally_step()` |
| **硬件检查** | `check_mxfp8_support()` 检查 SM >= 10.0 | `is_fp8_available()` **直接返回 True**（NPU 不检查） |

MindSpeed 的 `is_fp8_available()` 硬编码返回 `True`（`state_manager.py:58-59`），没有做 NPU 硬件能力检查。

### 2.2 FP8GlobalStateManager

| | TE 原生 | MindSpeed |
|---|---------|-----------|
| **全局状态** | `FP8_ENABLED`, `FP8_RECIPE`, `FP8_DISTRIBUTED_GROUP`, `AUTOCAST_DEPTH`, `IS_FIRST_FP8_MODULE` | 同名字段，额外增加 `FUSION_MATMUL`, `FP8_REUSE_QUANTIZED_WEIGHT` |
| **全局缓冲区** | `add_fp8_tensors_to_global_buffer()` — Delayed Scaling 需要注册 amax/scale 到全局 | **不存在** — MindSpeed 无全局缓冲区概念 |
| **amax 同步** | `reduce_and_update_fp8_tensors()` — 对 MXFP8 是 NOOP，对 Delayed 执行 all_reduce | `fp8_autocast_exit()` — 仅 Delayed 执行 `finally_step()` |

MindSpeed 的 StateManager 比 TE 简单很多（83 行 vs TE 的数百行），因为它不需要管理全局 amax buffer。

### 2.3 Recipe 体系

**TE 原生的抽象链**：

```
Recipe (MXFP8BlockScaling)
  → RecipeState (MXFP8BlockScalingRecipeState) — 无状态，只持有 dtype
    → Quantizer (MXFP8Quantizer) — 执行量化，管理 usage (rowwise/columnwise)
      → QuantizedTensor (MXFP8Tensor) — 存储 FP8 data + E8M0 scale_inv
```

**MindSpeed 的抽象链**：

```
RecipeScaling (MXFP8BlockScaling)     ← dataclass，仅持有 fp8_format
  → Recipe (MXFP8ScalingRecipe)       ← 直接包含 quantization() 方法
    → 调用 torch_npu 算子             ← 没有独立的 Quantizer 对象
      → MXFP8Tensor (Float8Tensor2D)  ← 存储 col_tensor + row_tensor
```

**关键差异**：

1. TE 有独立的 `MXFP8Quantizer` 对象，管理 `set_usage(rowwise, columnwise)` 语义
2. MindSpeed 没有 Quantizer 对象，`MXFP8ScalingRecipe.quantization()` 直接接收 `colwise`/`rowwise` 参数
3. TE 的 RecipeState 由 `BasicLinear.reset_recipe_state()` 在模块内创建并持有
4. MindSpeed 的 Recipe 由 `FP8Metadata.create_recipe()` 按需创建（懒初始化）

MindSpeed 支持 4 种 FP8 recipe：

| Recipe | 说明 | 对应 TE 概念 |
|--------|------|-------------|
| `delayed` | 延迟缩放 | `DelayedScaling` |
| `tensorwise` | 当前即算 per-tensor | `Float8CurrentScaling` |
| **`mxfp8`** | **每 32 元素 block scaling** | **`MXFP8BlockScaling`** |
| `blockwise` | 128x128 block scaling | `Float8BlockScaling` |

---

## 三、量化过程对比

### 3.1 量化调用链

**TE 原生**：

```
MXFP8Quantizer.quantize_impl()
  → tex.quantize(tensor, self)       ← C++ binding
    → quantize_mxfp8_kernel          ← CUDA kernel
      → 每 32 元素: amax → E8M0 scale → FP8
      → 单个 kernel 同时生成 rowwise + columnwise
```

**MindSpeed**：

```
MXFP8ScalingRecipe.quantization()
  ├─ rowwise && colwise:
  │    torch_npu.npu_dynamic_mx_quant_with_dual_axis()  ← 单算子，同时输出双向
  │    → (col_data, col_scale, row_data, row_scale)
  ├─ colwise only:
  │    torch_npu.npu_dynamic_mx_quant(axis=-1)
  └─ rowwise only:
       torch_npu.npu_dynamic_mx_quant(axis=-2)
```

**对应关系**：

| TE CUDA kernel | MindSpeed torch_npu 算子 | 功能 |
|----------------|------------------------|------|
| `quantize_mxfp8_kernel` (双向) | `npu_dynamic_mx_quant_with_dual_axis` | 同时生成 rowwise + columnwise |
| `quantize_mxfp8_kernel` (单向) | `npu_dynamic_mx_quant(axis=...)` | 单方向量化 |

TE 用单个 CUDA kernel + TMA + ping-pong buffer 完成双向量化。MindSpeed 依赖 `npu_dynamic_mx_quant_with_dual_axis` 一个 torch_npu 算子完成同样的事，内部实现由 CANN 提供。

### 3.2 MindSpeed 量化源码（`mxfp8_scaling_recipe.py`）

```python
class MXFP8ScalingRecipe(Recipe):
    def quantization(self, tensor, key, colwise, rowwise):
        tensor_2d = view_as_n_dim(tensor)
        fp8_dtype = self.quant_dtype
        mxfp8_tensor = MXFP8Tensor(fp8_dtype, tensor.shape, tensor.device, tensor.dtype, key=key)

        if rowwise and colwise:
            coly, col_scale, rowy, row_scale = self.run_quantizer(
                tensor_2d, key,
                torch_npu.npu_dynamic_mx_quant_with_dual_axis,
                dst_type=fp8_dtype,
            )
        elif colwise:
            coly, col_scale = self.run_quantizer(
                tensor_2d, key,
                torch_npu.npu_dynamic_mx_quant,
                axis=-1, dst_type=fp8_dtype,
            )
        elif rowwise:
            rowy, row_scale = self.run_quantizer(
                tensor_2d, key,
                torch_npu.npu_dynamic_mx_quant,
                axis=-2, dst_type=fp8_dtype,
            )

        # forward: x.col   @ w.col.T
        # dx     : g.col   @ w.row
        # dw     : g.row.T @ x.row
        mxfp8_tensor.set_row_data(rowy, row_scale, key == TensorKey.grads)
        mxfp8_tensor.set_col_data(coly, col_scale, key == TensorKey.weight)
        return mxfp8_tensor
```

---

## 四、Tensor 表示对比

### 4.1 数据结构

**TE 原生** `MXFP8Tensor`：

```
_rowwise_data:        (M, N)                        uint8 (FP8)
_rowwise_scale_inv:   (pad128(M), pad4(N/32))       uint8 (E8M0)
_columnwise_data:     (M, N)                        uint8 (FP8)
_columnwise_scale_inv:(pad4(M/32), pad128(N))       uint8 (E8M0)
_fp8_dtype:           E4M3 或 E5M2
_quantizer:           MXFP8Quantizer 引用
```

**MindSpeed** `MXFP8Tensor`（继承 `Float8Tensor2D`）：

```python
class QuantTensorMeta(NamedTuple):
    data: torch.Tensor     # FP8 量化数据
    scale: torch.Tensor    # E8M0 scale

class Float8Tensor2D:
    col_tensor: QuantTensorMeta   # columnwise (data + scale)
    row_tensor: QuantTensorMeta   # rowwise (data + scale)
    fp8_dtype                     # 量化目标 dtype
    origin_shape                  # 原始 tensor shape
    dtype                         # 原始高精度 dtype
    key                           # TensorKey (inputs/weight/grads)
```

**差异**：
- TE 有显式的 `pad128`/`pad4` 对齐逻辑，scale 存储有严格的 padding 规则
- MindSpeed 把 padding 交给 torch_npu 算子内部处理，Python 层不可见
- TE 持有 `_quantizer` 引用；MindSpeed 不持有
- MindSpeed 额外实现了 `release()` 方法，用完即释放 storage，显式管理显存

### 4.2 数据生命周期

```
npu_dynamic_mx_quant_with_dual_axis
    → (col_data, col_scale, row_data, row_scale)
        → 存入 MXFP8Tensor 的 col_tensor / row_tensor
            → GEMM 时按方向取出 (data, scale) 一起传入 npu_quant_matmul
                → GEMM 完成后 release() 释放 data 和 scale 的 storage
```

---

## 五、GEMM 调度对比

### 5.1 数据方向选择

**TE 原生**（cuBLAS 根据转置标志选择数据）：

| GEMM | TE layout | TE 数据选择 |
|------|-----------|-----------|
| Forward (y = W @ X) | 默认 TN | W: **rowwise**, X: **rowwise** |
| dgrad (dx = W^T @ dY) | NN | W: **columnwise**, dY: **rowwise** |
| wgrad (dw = X^T @ dY) | NT | X: **columnwise**, dY: **columnwise** |

**MindSpeed**（`constants.py:76-80`）：

```python
MATMUL_WISE_MAP = {
    MatmulKey.forward: (False, False),    # x.col @ w.col
    MatmulKey.dx:      (False, True),     # g.col @ w.row
    MatmulKey.dw:      (True, True),      # g.row @ x.row
}
# False = columnwise, True = rowwise
```

| GEMM | MindSpeed (is_rowwise) | MindSpeed 数据选择 |
|------|----------------------|-------------------|
| Forward | (False, False) | x: **col**, w: **col** |
| dgrad | (False, True) | g: **col**, w: **row** |
| wgrad | (True, True) | g: **row**, x: **row** |

**Forward 的数据方向不同**：TE 用 rowwise，MindSpeed 用 columnwise。不影响数学正确性，反映 NPU 和 CUDA GEMM kernel 对输入布局偏好不同。

### 5.2 GEMM 调用

**TE**：

```python
y, *_ = general_gemm(w, x, out_dtype=dtype, quantization_params=output_quantizer, ...)
# → cuBLAS MXFP8 GEMM，支持 NN/NT/TN layout
```

**MindSpeed**（`mxfp8_tensor.py:15-22`）：

```python
def quant_matmul(self, other: 'MXFP8Tensor', is_rowwise):
    x1, x1_scale = self.get_quant_data(is_rowwise[0])
    x2, x2_scale = other.get_quant_data(is_rowwise[1])
    output = torch_npu.npu_quant_matmul(
        x1, x2, x2_scale,
        pertoken_scale=x1_scale,
        output_dtype=self.dtype,
        scale_dtype=torch_npu.float8_e8m0fnu,
        pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
        group_sizes=[1, 1, 32]
    )
```

GEMM 输出始终是高精度（`output_dtype=self.dtype`，即 BF16），硬件内部 FP32 累加。

---

## 六、Backward 数据复用对比

### TE 原生

```
Forward 缓存:
  x_local: rowwise(可丢弃) + columnwise(backward wgrad 复用)
  w:       rowwise(可丢弃) + columnwise(backward dgrad 按需计算)

Backward 新计算:
  dy: rowwise(dgrad) + columnwise(wgrad)
```

TE 通过 `update_usage(rowwise_usage=False, columnwise_usage=True)` 在 forward 结束后释放不再需要的方向。

### MindSpeed

```python
# mxfp8_scaling_recipe.py:57-58
mxfp8_tensor.set_row_data(rowy, row_scale, key == TensorKey.grads)  # grads 做转置
mxfp8_tensor.set_col_data(coly, col_scale, key == TensorKey.weight) # weight 做转置
```

MindSpeed 没有 `update_usage` 机制，而是在 GEMM 执行完毕后通过 `release()` 立即释放 storage：

```python
def release(self, data, scale):
    if self.key == TensorKey.weight and FP8GlobalStateManager.is_weight_quantization_reuse_configured():
        return  # weight 复用时不释放
    data.untyped_storage().resize_(0)
    scale.untyped_storage().resize_(0)
```

---

## 七、通信路径对比

### 7.1 TE 原生

```
gather_along_first_dim(..., quantizer=input_quantizer)
  → 通信和量化可以合并
userbuffers overlap
```

TE 在 Python 层组合通信+量化，支持 userbuffers overlap。

### 7.2 MindSpeed — 两套路径

#### DefaultOps（默认路径）— 通信和计算分离

```python
# allgather_matmul: 先 HP 通信，再 FP8 计算
total_input = torch.empty(...)
torch.distributed._all_gather_base(total_input, input_)   # ← BF16 all-gather
output, _, _ = fp8_matmul(total_input, weight, ...)        # ← 再量化 + GEMM

# matmul_reduce_scatter: 先 FP8 计算，再 HP 通信
output_, _, _ = fp8_matmul(input_, weight, ...)            # ← 量化 + GEMM，输出 BF16
torch.distributed._reduce_scatter_base(output, output_)   # ← BF16 reduce-scatter
```

**通信全部走高精度（BF16），不是 MXFP8。**

#### Mc2Ops（MC2 路径）— 通信和计算融合，MXFP8 直接通信

```python
# fp8_all_gather_matmul:
inputs = fp8_meta.quantization(...)    # 先量化为 MXFP8
inputs.all_gather_matmul(weight, ...)  # ← 调用 MXFP8Tensor 的方法

# MXFP8Tensor.all_gather_matmul 内部：
def all_gather_matmul(self, other, bias, fp8_meta, key):
    # 1. row 方向：FP8 data 和 scale 分别做 all-gather（给 backward 用）
    _, row_data  = all_gather_along_dim(row_data)     # ← FP8 data all-gather
    _, row_scale = all_gather_along_dim(row_scale)    # ← E8M0 scale all-gather

    # 2. col 方向：融合 all-gather + GEMM
    output = npu_all_gather_quant_mm(
        self.col_tensor.data, x2,          # ← FP8 data 直接参与通信
        x1_scale=self.col_tensor.scale,
        group_sizes=[1, 1, 32],
    )

    # 3. 拼装 gather 后的完整 MXFP8Tensor
    gather_out = MXFP8Tensor(...)
    gather_out.set_row_data(row_data, row_scale, ...)
    return output, gather_out
```

#### 通信对比总结

| | DefaultOps | Mc2Ops |
|---|-----------|--------|
| **all-gather 通信内容** | BF16 原始 tensor | **FP8 data 直接通信** |
| **all-gather 方式** | HP all-gather → 量化 → GEMM | all-gather + GEMM 融合（`npu_all_gather_quant_mm`） |
| **reduce-scatter 通信内容** | BF16 GEMM 输出 | BF16 GEMM 输出（HP） |
| **reduce-scatter 方式** | GEMM → HP reduce-scatter | GEMM + reduce-scatter 融合（`npu_quant_mm_reduce_scatter`） |
| **启用条件** | 默认路径 | `--use-ascend-mc2` 且 `--fp8-recipe mxfp8` |

MC2 路径下 all-gather 阶段是 MXFP8 直接通信（省带宽），reduce-scatter 阶段通信的是 GEMM 的高精度输出。

---

## 八、梯度累加路径

### 8.1 Dense Linear

```python
# linear.py:559-566
if ctx.fp8_enable:
    if ctx.gradient_accumulation_fusion and isinstance(ctx.fp8_meta.fp8_recipe, MXFP8BlockScaling):
        # 融合路径：GEMM + 原地累加到 FP32 main_grad
        fp8_matmul_add(weight_param.main_grad, grad, total_input, ctx.fp8_meta)
    else:
        # 标准路径：GEMM 输出高精度 dw
        grad_weight, _, _ = fp8_matmul(grad, total_input, ctx.fp8_meta, MatmulKey.dw)
```

`fp8_matmul_add` 内部调用 `npu_add_quant_matmul_`，语义等价于：

```
main_grad(FP32) += GEMM(dy_mxfp8, x_mxfp8)  # GEMM 输出天生是 HP，累加到 FP32
```

### 8.2 MoE Grouped MatMul

两个调用点：

| 路径 | 累加目标 | 精度 |
|------|---------|------|
| `grouped_matmul_util.py` (MCore) | `weight_param.main_grad` | **FP32** |
| `gmm_mxfp8.py` (TE 模块) | `weight_param.grad` | 跟 weight 同 dtype（通常 BF16） |

MCore 路径用 `main_grad`（FP32）与 Dense Linear 一致。`gmm_mxfp8.py` 用 `.grad`，在多 micro-batch 累加场景下精度略低。

### 8.3 GEMM 输出精度

所有 GEMM 输出天生是高精度，与 TE 一致：

```python
output = torch_npu.npu_quant_matmul(..., output_dtype=x.dtype, ...)  # BF16
```

FP8 输入 → 硬件内 FP32 累加 → 以 output_dtype 输出。不存在"FP8 梯度累加"。

---

## 九、torch_npu 原生算子清单

### 9.1 量化算子

| 算子 | 签名 | 用途 |
|------|------|------|
| `npu_dynamic_mx_quant` | `(tensor, axis, dst_type) → (data, scale)` | 单轴 MXFP8 量化，per-32-block E8M0 scale |
| `npu_dynamic_mx_quant_with_dual_axis` | `(tensor, dst_type) → (col_data, col_scale, row_data, row_scale)` | 双轴 MXFP8 量化，一次产出行+列 |
| `npu_grouped_dynamic_mx_quant` | `(tensor, group_list, round_mode, dst_type, blocksize) → (data, scale)` | 分组 MXFP8 量化（MoE 专用） |
| `npu_dynamic_quant` | `(tensor, dst_type, quant_mode) → (data, scale)` | Per-tensor 动态量化（tensorwise recipe） |
| `npu_quantize` | `(tensor, scale, zero_points, dtype, axis) → data` | 静态 scale 量化（delayed recipe） |
| `npu_dynamic_block_quant` | `(tensor, dst_type, row_block_size, col_block_size) → (data, scale)` | 固定 block 维度量化（blockwise recipe） |

### 9.2 矩阵乘法算子

| 算子 | 用途 | MXFP8 特有参数 |
|------|------|---------------|
| `npu_quant_matmul` | 量化矩阵乘，核心 GEMM | `scale_dtype=float8_e8m0fnu`, `group_sizes=[1,1,32]` |
| `npu_add_quant_matmul_` | 量化矩阵乘 + 梯度原地累加 | 同上 |
| `npu_grouped_matmul` | 分组量化矩阵乘（MoE） | `scale_dtype=float8_e8m0fnu`, `split_item=3` |
| `npu_add_quant_gmm_` | 分组量化矩阵乘 + 梯度原地累加（MoE） | 同上 |

### 9.3 通信融合算子

| 算子 | 用途 | MXFP8 特有参数 |
|------|------|---------------|
| `npu_all_gather_quant_mm` | All-Gather + 量化矩阵乘融合 | `x1_scale_dtype=float8_e8m0fnu`, `group_sizes=[1,1,32]` |
| `npu_quant_mm_reduce_scatter` | 量化矩阵乘 + Reduce-Scatter 融合 | 同上 |

### 9.4 数据类型常量

| 常量 | 含义 |
|------|------|
| `torch_npu.float8_e8m0fnu` | E8M0 dtype，MXFP8 的 scale dtype |
| `torch_npu.hifloat8` | 华为自有 HiFloat8 格式 |

### 9.5 各 Recipe 使用的算子对比

```
                    量化算子                              GEMM 算子          group_sizes
───────────────────────────────────────────────────────────────────────────────────────
delayed          npu_quantize                           npu_quant_matmul     (无)
tensorwise       npu_dynamic_quant                      npu_quant_matmul     (无)
mxfp8            npu_dynamic_mx_quant                   npu_quant_matmul     [1, 1, 32]
                 npu_dynamic_mx_quant_with_dual_axis
blockwise        npu_dynamic_block_quant                npu_quant_matmul     [1, 128, 128]
───────────────────────────────────────────────────────────────────────────────────────
MoE (mxfp8)     npu_dynamic_mx_quant                    npu_grouped_matmul  [1, 1, 32]
                 npu_grouped_dynamic_mx_quant            npu_add_quant_gmm_
```

---

## 十、FP8 数据格式支持

| 格式组合 | inputs | weights | gradients |
|---------|--------|---------|-----------|
| E4M3 | E4M3 (±448) | E4M3 (±448) | E4M3 (±448) |
| HYBRID | E4M3 (±448) | E4M3 (±448) | E5M2 (±57344) |
| HIF8 | HIF8_15 (±15) | HIF8_15 (±15) | HIF8_224 (±224) |

- MXFP8 recipe 可与 E4M3、HYBRID 搭配
- HIF8 仅支持 tensorwise recipe
- MXFP8 默认用 E4M3（block-level scale 提供了足够的动态范围适应，E4M3 的更高精度更有利）

---

## 十一、Shape 约束对比

**TE 原生**：

```python
# MXFP8Quantizer.is_quantizable()
def is_quantizable(self, inp):
    if inp.ndim < 2: return False
    if inp.shape[-1] % 32 != 0: return False
    if math.prod(inp.shape[:-1]) % 32 != 0: return False
    return True
```

**MindSpeed**：没有在 Python 层做显式的 shape 检查。如果不满足 32 对齐，会在 torch_npu 算子层报错。

---

## 十二、分布式训练对比

### MXFP8 不需要 amax 同步

```
Delayed Scaling:
  GPU 0: amax=1.5 ─┐
  GPU 1: amax=2.0 ─┼─► all_reduce(MAX) ─► amax=2.0 ─► scale
  GPU 2: amax=1.8 ─┘
  → 需要跨 GPU 通信来同步 scale

MXFP8 Block Scaling:
  GPU 0: 每个 block 即时计算自己的 scale
  GPU 1: 每个 block 即时计算自己的 scale
  → 各 GPU 独立计算 scale，无需通信
```

TE 和 MindSpeed 在这一点上一致。

### MC2 约束

**MXFP8 是 MindSpeed 中唯一支持 MC2 的 FP8 recipe**：

```python
# transformer_engine_basic.py:74
if args.use_ascend_mc2 and args.fp8 and args.fp8_recipe != 'mxfp8':
    raise AssertionError('MC2 is supported only by the mxfp8 recipe in fp8.')
```

---

## 十三、MindSpeed 缺失或简化的部分

| TE 原生能力 | MindSpeed 状态 | 说明 |
|------------|--------------|------|
| `MXFP8Quantizer` 独立对象 | **无** | 没有独立的 Quantizer，usage 由调用参数直接传入 |
| `MXFP8Quantizer.set_usage()` | **无** | 无 usage 语义管理 |
| `MXFP8Quantizer.is_quantizable()` | **无** | 不做 Python 层 shape 校验 |
| `MXFP8TensorStorage` (padding 对齐) | **透传** | padding 逻辑在 torch_npu 算子内部 |
| Dequantization (`tex.dequantize()`) | **无独立接口** | 反量化在 GEMM kernel 内完成 |
| `fp8_dpa` / `fp8_mha` | **不支持** | MXFP8 不覆盖 Attention 内部 |
| Activation caching (双向量化中缓存中间结果) | **依赖算子** | `npu_dynamic_mx_quant_with_dual_axis` 内部实现 |
| TMA 异步加载 + ping-pong | **不适用** | NPU 架构不同，由 CANN 调度 |
| cuBLAS MXFP8 GEMM layout 选择 | **不适用** | NPU 通过 `npu_quant_matmul` 参数约定 |
| `check_mxfp8_support()` 硬件检查 | **硬编码 True** | 不检查 NPU 是否支持 MXFP8 |
| Block 对齐 / shard 边界 fallback | **无显式检查** | TE 在 shard 破坏 32 对齐时退回 HP |

---

## 十四、MindSpeed 额外增加的能力

| 能力 | 位置 | 说明 |
|------|------|------|
| Weight 量化复用 | `reuse.py` + `state_manager.py` | optimizer step 内复用已量化的 weight |
| CPU 对比验证 | `mxfp8_tensor_cpu.py` | `--te-comparison-with-cpu` 在线校验 NPU 量化精度 |
| BF16 对比验证 | `__init__.py:169` | `--te-comparison-with-bf16` 对比 FP8 vs BF16 精度损失 |
| 梯度累加融合 | `fp8_matmul_add()` | `npu_add_quant_matmul_` 融合 GEMM + grad 累加 |
| MoE Grouped MatMul | `gmm_mxfp8.py` | `npu_grouped_matmul` + `npu_grouped_dynamic_mx_quant` |
| FSDP 集成 | `fsdp/quantization/` | `MXLinear` + `MXTensor` 独立于 TE 模块路径 |
| MC2 支持 | 验证 + MC2 Ops | MXFP8 是唯一支持 MC2 的 FP8 recipe |
| HIF8 格式 | `constants.py` | 华为自有 `hifloat8` 格式（仅 tensorwise） |
| 显存显式管理 | `Float8Tensor2D.release()` | GEMM 后立即释放 quantized data storage |

---

## 十五、关键代码路径索引

| 功能 | 文件 |
|------|------|
| autocast 入口 | `te/pytorch/fp8/fp8.py` |
| 全局状态管理 | `te/pytorch/fp8/state_manager.py` |
| FP8 元数据 + Recipe 创建 | `te/pytorch/fp8/metadata.py` |
| Recipe 基类 | `te/pytorch/fp8/recipes/recipe.py` |
| MXFP8 Recipe + MatMul | `te/pytorch/fp8/recipes/mxfp8_scaling_recipe.py` |
| Delayed Scaling Recipe | `te/pytorch/fp8/recipes/delayed_scaling_recipe.py` |
| Tensorwise Recipe + MatMul | `te/pytorch/fp8/recipes/current_scaling_recipe.py` |
| Blockwise Recipe + MatMul | `te/pytorch/fp8/recipes/float8_block_scaling_recipe.py` |
| FP8 格式 / Recipe 枚举 / MatmulKey | `te/pytorch/fp8/constants.py` |
| fp8_matmul / fp8_matmul_add | `te/pytorch/fp8/__init__.py` |
| Float8Tensor / Float8Tensor2D | `te/pytorch/fp8/tensor/float8_tensor.py` |
| MXFP8Tensor (NPU) | `te/pytorch/fp8/tensor/mxfp8_tensor.py` |
| MXFP8TensorCpu (CPU 对比) | `te/pytorch/fp8/tensor/mxfp8_tensor_cpu.py` |
| Float8BlockTensor | `te/pytorch/fp8/tensor/float8_block_tensor.py` |
| Weight 量化复用 | `te/pytorch/fp8/reuse.py` |
| TEColumnParallelLinear / TERowParallelLinear | `te/pytorch/module/linear.py` |
| DefaultOps (通信分离) | `te/pytorch/module/ops/default_ops.py` |
| Mc2Ops (通信融合) | `te/pytorch/module/ops/mc2_ops.py` |
| MoE MXFP8 GroupedMatMul | `ops/gmm_mxfp8.py` |
| MoE MCore GroupedMatMul | `core/transformer/moe/grouped_matmul_util.py` |
| FSDP MXFP8 Config | `fsdp/quantization/mxfp8_config.py` |
| FSDP MXLinear | `fsdp/quantization/mx_formats/mx_linear.py` |
| FSDP MXTensor | `fsdp/quantization/mx_formats/mx_tensor.py` |
| Feature 注册 + 参数验证 | `features_manager/megatron_basic/transformer_engine_basic.py` |
| FP8 Context (Megatron 入口) | `core/fp8_utils.py` |

---

## 十六、总结

MindSpeed 的 TE 实现是一套**完全独立于 TE CUDA 后端的重新实现**，保留了 TE 的接口约定（autocast、recipe、Format 等），核心特征：

1. **砍掉了 C++/CUDA 层**，所有计算通过 torch_npu 算子完成（10 个量化/计算算子 + 2 个通信融合算子）
2. **简化了抽象层**，去掉了独立的 Quantizer 对象和 RecipeState 中间层
3. **简化了状态管理**，不需要全局 amax buffer（MXFP8 本身无状态），StateManager 仅 83 行
4. **增加了 NPU 特有能力**：weight 复用、精度在线校验、MC2 集成、MoE GMM 融合、显存显式管理
5. **数学等价但布局不同**：forward GEMM 的 rowwise/columnwise 选择与 TE 相反，反映 NPU 硬件偏好
6. **通信双路径**：DefaultOps 走 BF16 通信，Mc2Ops 走 MXFP8 直接通信 + 计算融合
