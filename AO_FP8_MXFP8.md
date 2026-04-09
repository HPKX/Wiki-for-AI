# FP8 / MXFP8 机制、训练流程与 TorchTitan/TorchAO/FSDP2 关系梳理

## 0. 阅读指引

- 第 1 节说明 FP8 训练的收益及其主要来源。
- 第 2 节先以 TorchAO stable float8 为默认主线，说明 FP8 训练在机制层面的核心变化；随后单独讨论 MXFP8 作为更细粒度 microscaling 变体的差异。
- 第 3 节中，`3.1` 到 `3.4` 说明 stable float8 的实现路径，`3.5` 到 `3.7` 说明 MXFP8 的实现路径，`3.8` 到 `3.10` 为并排对照。

## 1. FP8 低精训练的收益及收益来源

### 1.1 收益成立的前提

- FP8 训练的核心价值，不在于将模型永久压缩为低 bit 存储，而在于降低训练关键路径上 `Linear/GEMM` 的执行成本；对大模型预训练而言，这一路径通常也是吞吐提升的主要来源。
- 在多卡场景下，这一单卡计算收益能否转化为端到端收益，还取决于通信路径和框架额外开销是否得到同步优化。

### 1.2 收益来源的拆解

| 收益来源 | 作用对象 | 是否主要来源 | 说明 |
| --- | --- | --- | --- |
| 低精度 GEMM | `Linear` 前向与反向的矩阵乘法 | 是 | 端到端加速的第一来源，也是 FP8 训练最核心的价值所在 |
| 低精度权重通信 | `float8 all-gather` | 条件性 | 多卡、大模型、通信占比较高时更重要 |
| `torch.compile` 摊薄开销 | cast / quantize / scale / dispatch | 获得最佳/竞争性性能的默认前提 | 决定低精度 GEMM 的收益能否真正转化为端到端收益 |

`torch.compile` 在这里主要承担两类作用：一是将 `amax/scale` 计算、cast、reshape 以及 tensor subclass dispatch 等外围逻辑收进编译图，减少 Python 与 dispatcher 开销；二是在可能时融合低精 GEMM 前后的 pointwise kernel。`scaled_mm` / rowwise / MX kernel 仍是核心算子，但 compile 决定这些算子周围的额外开销能否被摊薄。按照 TorchTitan 对 float8 / MXFP8 的官方说明，`torch.compile` 可视为获得最佳/竞争性性能的默认前提。

### 1.3 代表性性能结果

#### `Supercharging Training using float8 and FSDP2`

- 在 H100 集群实验中，文中给出的代表性结果包括：8B 约提升 `28%`，70B 约提升 `50%`，405B 约提升 `52%`；对应环境覆盖 `128 / 256 / 512 H100`，其中 `405B` 路径结合了 `TP4 + FSDP2`，序列长度统一为 `8K`。
- 同一篇文章还给出一个较粗粒度但重要的结论：在 compute 已经切换到 float8 之后，继续引入 `float8 all-gather` 还能额外带来约 `5%` 的提升。
- 结果表明，随着模型规模增大，FP8 更容易摊薄额外开销并转化为明显吞吐收益；同时，通信路径的低精度化本身也会对端到端收益产生独立贡献。

#### `Quantized Training`

- 在 H100 单机 8 卡 Llama3-8B 场景下，`tensorwise + float8 all-gather` 相比 BF16 提升约 `25.03%`，`rowwise + bf16 all-gather` 提升约 `10.05%`。
- 结果表明，不同 recipe 在吞吐与稳定性之间存在清晰权衡。

### 1.4 适用边界与限制

- FP8 并不保证所有 shape 都更快。小 shape 下，量化、scale 计算和 tensor subclass 分发开销可能超过 GEMM 本身的节省。
- 不同 recipe 在性能和稳定性之间存在明确权衡，因此“更低精度”并不等于“端到端一定更优”。

## 2. FP8 低精训练的核心思想与机制

本节包括 stable float8 与 MXFP8 两部分内容。

- `2.1` 到 `2.4` 讨论 TorchAO stable float8 的默认机制主线。
- `2.5` 讨论 MXFP8 这一 microscaling 变体，并在必要处与前文对照。

### 2.1 训练里真正改变了哪些核心计算

可将 BF16 baseline 近似表示为：

```text
output = input_bf16 @ weight_bf16.T
grad_input = grad_output_bf16 @ weight_bf16
grad_weight = input_bf16.T @ grad_output_bf16
```

相应地，FP8 训练可近似表示为：

```text
output = to_fp8(input) @ to_fp8(weight.T)
grad_input = to_fp8(grad_output) @ to_fp8(weight)
grad_weight = to_fp8(input.T) @ to_fp8(grad_output)
```

进一步抽象后，一个 `Linear` 层在一次前向和反向中涉及三次核心矩阵乘法：

```text
1. forward:   output      = input       @ weight^T
2. backward:  grad_input  = grad_output @ weight
3. backward:  grad_weight = input^T     @ grad_output
```

对应结论如下：

- TorchAO stable float8 training 目前重点支持的是 `torch.nn.Linear` 对应的 GEMM 路径。
- 改变的是这三次 GEMM 的 Tensor 表示和执行路径，而不是把整个模型训练状态永久存成 8-bit。
- 输出、loss、optimizer step 仍然处于正常的高精度训练语义中。

### 2.2 FP8 不是裸 dtype，而是 `qdata + scale + kernel`

当这三次 GEMM 的计算被低精度化时，需要进一步区分两层概念：

- 基础 dtype：例如 `torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch.float8_e8m0fnu`（MXFP8 scale 专用，见 `2.5`）。
- 训练里真正使用的低精度表示：通常是“低精度数据 + scale + 专用 kernel”的组合，而不是简单的 `tensor.to(float8)`。

在工程实现中，`E4M3` 与 `E5M2` 往往承担不同角色：前者精度更高，更常用于 `input / weight`；后者范围更大，更常用于 `grad_output`。TorchAO stable float8 的默认 tensorwise 配置即采用这一分工，而 `rowwise` recipe 通常统一使用 `E4M3`。

在训练实现中，“FP8”对应以下执行表示：

- `qdata`：低精度数据本体。
- `scale`：用于将高精度数值范围映射到低精度表示范围的缩放因子。
- `kernel`：能够识别 `qdata + scale` 的矩阵乘法执行路径，例如 `torch._scaled_mm`。

### 2.3 为什么一定要有 scale

裸 FP8 的表示范围和精度通常不足以直接覆盖训练中的 activation / weight / gradient 分布，尤其在存在 outlier 时，直接 cast 很容易造成信息损失。

因此，FP8 训练依赖 scale 机制：

- 先根据高精度 tensor 当前的数值范围计算 scale。
- 再将高精度 tensor 映射到 FP8 的可表示范围内。
- matmul kernel 消费的并不是单独的 FP8 dtype，而是 `qdata + scale`。

本文中，**scale 的定义域**指一个 scale 值所覆盖的原始高精度数据范围。对于 tensorwise，定义域为整个 tensor；对于 rowwise（axiswise），定义域为沿指定 axis 的一个切片；对于 MXFP8，定义域为内存中连续 32 个元素构成的 block。

在分布式场景下，跨 rank 是否需要对齐 `amax / scale`，核心取决于 scale 的定义域是否跨越 rank 分片边界。若未跨越，本地即可直接计算对应 scale，并天然保持与单卡一致的量化语义。

若跨越，则其量化语义需要保持与单卡完整逻辑 tensor 一致，即保持 `single-device semantics`。此时各 rank 不能仅依据本地 shard 独立量化，而需要通过显式 collective 或等价的分布式 reduction 语义对齐量化语义。这一判断不由具体通信原语决定。

FSDP 与 TP 是这一原则的两类典型实例：前者通过显式 collective 或预计算 scale 对齐全局 `amax`，后者通过 DTensor 的 reduction 语义隐式处理。各 recipe 与并行策略组合下 scale 通信是否发生、通过何种机制发生，具体分析见 `3.4`。

### 2.4 stable float8 recipe：scale 粒度与权衡

在明确了“GEMM 是主要优化对象”以及“低精度表示由 `qdata + scale + kernel` 组成”之后，stable float8 各 recipe 的区别主要体现在 scale 粒度以及哪些路径保留高精度保护。

TorchAO 当前 stable float8 的常见路线如下。

| 方案 | scale 粒度 | 核心机制 | 数值稳定性 | 性能倾向 |
| --- | --- | --- | --- | --- |
| `tensorwise` | 每个 tensor 1 个 scale | 整块数据共用一个 scale | 最低，对 outlier 最敏感 | 通常最高 |
| `rowwise` | 每行 1 个 scale | 将 outlier 影响限制在行内 | 中高 | 中等 |
| `rowwise_with_gw_hp` | 前向/部分反向按 rowwise，`grad_weight` 保持高精度 | 保护最敏感的 `grad_weight` 路径 | 最高 | 最低 |

上述路线对应同一性能-稳定性曲线上的不同配置点：

- `tensorwise` 更偏向吞吐性能。
- `rowwise` 更偏向 outlier 控制与 loss parity 的保持。
- `rowwise_with_gw_hp` 是更保守的稳定性优先方案。

### 2.5 MXFP8：作为 FP8 主线的进一步演进

MXFP8 并非与 FP8 完全平行的一套独立算法，其与 FP8 的关系如下：

- FP8 是低精度浮点训练的总体类别。
- scaled FP8 是常见的工程实现形态，即低精度数据需要与 scale 配合使用，才能在训练中保持可接受的数值稳定性。
- MXFP8 可视为 scaled FP8 沿更细粒度 microscaling 与更强硬件原生性方向的进一步演进，其价值更依赖对 MX 原生表示的硬件与 kernel 支持。

除 scale 粒度外，MXFP8 与当前 TorchAO stable float8 还在数值表示、量化粒度、反向实现路径以及硬件耦合方式等方面存在差异。

| 维度 | 常规 FP8（TorchAO stable float8） | MXFP8 | 工程含义 |
| --- | --- | --- | --- |
| scale 粒度 | 通常采用 `tensorwise` 或 `rowwise/axiswise` | 固定为每 32 个元素共享 1 个 scale | MXFP8 对局部 outlier 的约束更细，不再由整行或整个 tensor 共享同一动态范围 |
| scale 表示 | scale 通常按 `amax -> finfo(fp8).max / amax` 动态计算得到；`rowwise` recipe 默认采用 round 到 2 的幂 | scale 直接编码为 `float8_e8m0fnu`，即 OCP MX 规范中的 `E8M0` microscale | MXFP8 将“scale 的存储方式与 kernel 消费方式”纳入格式定义，而不是仅将其作为运行时附带的浮点缩放系数 |
| 数据组织 | 更接近“FP8 数据 + 动态 scale + scaled_mm kernel”的通用工程方案 | 采用更具体的 MX 格式：`FP8 element + E8M0 scale + block size 32` | MXFP8 更贴近硬件原生格式；常规 FP8 则更接近通用训练抽象 |
| 反向路径 | 根据 recipe 的不同，可能采用 tensorwise / rowwise，也可能像 `rowwise_with_gw_hp` 那样对 `grad_weight` 保留高精度保护 | 前向采用 dim0 量化，反向显式引入 dim1 colwise 量化；`grad_input` 与 `grad_weight` 均依赖额外的 cast kernel | MXFP8 的 backward 实现复杂度更高，其中 dim1 quantize 是性能关键路径 |
| kernel / 硬件路径 | 主要围绕 `torch._scaled_mm`、tensorwise/rowwise kernel 以及相对成熟的 float8 训练路径展开 | 硬件加速路径明确对应 Blackwell SM100+ 的 cuBLAS MX GEMM，并要求 scale 采用 blocked/swizzled layout | MXFP8 能否转化为端到端收益，更依赖支持 MX 表示的硬件路径与专用 kernel 的成熟度 |
| 成熟度与定位 | 当前是 TorchAO / TorchTitan 中更成熟、也更容易与 FSDP2、float8 all-gather、`torch.compile` 组合的主线 | 更偏向面向 MX 原生硬件路径的下一阶段演进，API 与 kernel 仍处于快速变化阶段 | 工程实施顺序通常为先验证 stable float8 路线，再进行 MXFP8 评估 |

上表中与 scale 相关的差异，其工程含义可进一步概括为两点：

- 常规 FP8 的核心问题主要在于一个 scale 覆盖多大范围的数据；`rowwise` 路线通过更细粒度的 scale 以及 `power-of-2 scale` 控制舍入误差与局部 outlier。
- MXFP8 则不仅继续缩小 scale 的覆盖范围，还将 scale 的编码、layout 与 kernel 输入契约一并纳入格式定义。因此，它不能简单理解为对 `rowwise` 的进一步细分，其收益也更依赖底层 kernel 真正消费 `FP8 + E8M0 + blocked layout` 这一表示。

两者关系如下：

- 常规 FP8 是当前训练系统中相对成熟的 scaled FP8 主线。
- MXFP8 是该主线沿更细粒度 microscaling 与更强硬件原生性方向的进一步演进。

## 3. Torch 中的两条实现主线

`3.1` 到 `3.4` 展开 stable float8 实现路径（其中 `3.1` 到 `3.3` 为单卡路径，`3.4` 为分布式路径与 scale 通信）；`3.5` 到 `3.7` 展开 MXFP8 实现路径；`3.8` 到 `3.10` 为并排对照。

### 3.1 stable float8 主线：实现总览

`3.1` 到 `3.4` 讨论当前更成熟、也是前文默认采用的 TorchAO stable float8 主线（其中 `3.1` 到 `3.3` 为单卡路径，`3.4` 为分布式路径与 scale 通信）；`3.5` 之后讨论 MXFP8。

在 Torch 栈中，stable float8 主线如下：

```text
TorchTitan
  -> torchao.float8 recipe
  -> FSDP2 / DTensor
  -> float8 all-gather
  -> torch.compile
  -> scaled_mm / rowwise kernels
```

各组件职责如下：

- TorchTitan：训练入口、recipe、配置与工程编排。
- TorchAO：低精度表示、量化逻辑与算子路径。
- FSDP2 / DTensor：参数分片、all-gather、reduce-scatter 与参数生命周期管理。
- `torch.compile`：将 `amax/scale` 计算、cast、reshape 与 tensor subclass dispatch 收进编译图，并在可能时融合低精 GEMM 前后的 pointwise kernel，以摊薄额外开销。

### 3.2 模块替换：`nn.Linear -> Float8Linear`

TorchAO stable float8 training 的入口如下：

```python
def convert_to_float8_training(module, *, module_filter_fn=None, config=None):
    if config is None:
        config = Float8LinearConfig()
    from_float = lambda m: Float8Linear.from_float(m, config=config)
    return swap_linear_layers(module, from_float, module_filter_fn=module_filter_fn)

@classmethod
def from_float(cls, mod, config=None):
    with torch.device("meta"):
        new_mod = cls(mod.in_features, mod.out_features, bias=False, config=config)
    new_mod.weight = mod.weight
    new_mod.bias = mod.bias
    if config.enable_fsdp_float8_all_gather:
        new_mod.weight = torch.nn.Parameter(
            WeightWithDynamicFloat8CastTensor(...),
            requires_grad=new_mod.weight.requires_grad,
        )
    return new_mod
```

该入口包含以下要点：

- 入口的本质是把模块中的 `nn.Linear` 替换成 `Float8Linear`。
- 权重默认仍以高精度形式持久保存。
- 只有在启用 `enable_fsdp_float8_all_gather` 时，权重才会被包成能够参与 FSDP hook 的 tensor subclass。

### 3.3 计算表示：`Float8TrainingTensor` 与 `torch._scaled_mm`

`Float8Linear` 的前向进入一条“按需量化、低精度 GEMM、输出回到高精度语义”的路径：

```python
def forward(self, input: torch.Tensor) -> torch.Tensor:
    if torch.is_autocast_enabled():
        input = input.to(torch.get_autocast_gpu_dtype())
    output = matmul_with_hp_or_float8_args.apply(
        input,
        self.weight.t(),
        self.linear_mm_config,
        self.config,
    )
    if self.bias is not None:
        output = output + self.bias.to(output.dtype)
    return output
```

在这条路径中，关键不在函数名，而在对象之间的关系：

- 输入和权重会在需要时被转成 `Float8TrainingTensor`。
- `Float8TrainingTensor` 内部保存的是 `qdata + scale + metadata`。
- 后续 `torch.mm(...)` 会经由 `__torch_dispatch__` 分发到 `float8_ops.py`，最终进入 `torch._scaled_mm`。

因此，TorchAO 并非将张量简单 cast 为 FP8 后再执行普通 matmul，而是通过 tensor subclass 绑定低精度表示与执行路径。

### 3.4 分布式路径与 scale 通信

如果仅优化单卡 GEMM，整体收益仍不完整。对于大模型训练，更关键的是使低精度表示进入分布式参数生命周期。

这一点在 TorchAO + FSDP2 中主要体现为：

- 初始化后，参数会成为分片的 `DTensor`。
- `DTensor._local_tensor` 仍可保留 float8-aware 的权重 wrapper (`WeightWithDynamicFloat8CastTensor`)。
- 在 all-gather 前，hook 将本地高精度 shard 转成 `qdata + scale`。
- 在 all-gather 后，hook 再将 gathered `qdata + scale` 重组为当前 GEMM 可消费的 `Float8TrainingTensor`。

对应的关键 hook 形态如下：

```python
def fsdp_pre_all_gather(self, mesh):
    if self._precomputed_scale is not None:
        float8_training_tensor = hp_tensor_and_scale_to_float8(...)
    else:
        float8_training_tensor = hp_tensor_to_float8_dynamic(..., reduce_amax=True, device_mesh=mesh)
    return (float8_training_tensor._data,), (float8_training_tensor._scale,)

def fsdp_post_all_gather(self, all_gather_outputs, metadata, param_dtype, *, out=None):
    (data,) = all_gather_outputs
    (scale,) = metadata
    return Float8TrainingTensor(data, scale, param_dtype, self._linear_mm_config, gemm_input_role=GemmInputRole.WEIGHT), (data,)
```

该路径的核心通信语义是：先在本地将高精度 weight shard 转为 `qdata + scale`，再 all-gather 这一低精表示；而不是先 all-gather BF16 权重、再在本地 cast。因此通信 payload 即为后续 GEMM 可消费的低精度数据，通信量相应减少。

上述 FSDP hook 中使用的 scale 可以进一步预计算。`precompute_float8_dynamic_scale_for_fsdp(model)` 通常在 `optimizer.step()` 之后调用，用于统一计算并缓存下一轮 cast 所需的 scale，而不是提前将权重永久转成 FP8。其主要作用包括：

- 减少每层单独动态计算 scale 的开销。
- 将多处同步合并成一次更可控的 collective。
- 让下一轮 `fsdp_pre_all_gather()` 可以直接复用 `_precomputed_scale`，避免在热路径里再做一轮 amax 归约。

`TP / SP` 路径的 scale 对齐方式与 `FSDP2` 不同：权重以 DTensor 永久分片，scale 计算中的 `torch.max` / `torch.amax` 通过 DTensor dispatch 自动处理跨 rank reduction，无需像 FSDP 路径那样显式传入 `reduce_amax`。

对 `rowwise` recipe，当前主线选择保持高精度 operand 通信（不启用 float8 all-gather）。scale 定义域从整个 tensor 收缩为 axiswise 的局部范围，其在 TP 下是否需要跨 rank 通信取决于具体的 Shard 方向，详见下方分析。

#### 各 recipe 与并行策略下的 scale 通信分析

综合 `2.3` 中的 `single-device semantics` 原则以及上述实现路径，scale/amax 是否需要跨 rank 通信可按两个判据推导：

1. **量化时机**：量化发生在 distributed communication 之前还是之后。若在通信之后，各 rank 已持有完整数据，scale 本地计算即可。
2. **TP 下的 reduce 维度**：TP 场景下权重永久分片，不经过 all-gather，scale 计算通过 DTensor dispatch 执行。当 amax 的 reduce 维度与 Shard 维度一致时，DTensor 触发跨 rank all-reduce；正交时本地完成。

由此推导各组合下 weight scale 通信的结论如下：

| recipe | FSDP | TP |
| --- | --- | --- |
| tensorwise（float8 all-gather） | 需要：量化在 all-gather 前，通过 `reduce_amax` 或 `precompute` 对齐全局 amax | 需要：全维度 reduce 必然涉及 Shard 维度，DTensor 自动 all-reduce |
| rowwise | 不需要：不支持 float8 all-gather，BF16 gather 后本地量化 | 视 axiswise_dim 与 Shard 维度关系而定 |

补充说明：

- 若 tensorwise 不启用 float8 all-gather，FSDP 路径同样不需要 amax 通信（BF16 gather 后本地量化），但也失去低精通信收益。
- 在 TP + DP 复合场景下，`fsdp_pre_all_gather` 中的 `reduce_amax` 仅覆盖 FSDP mesh（DP 维度），各 TP shard 的 scale 彼此独立；`precompute_float8_dynamic_scale_for_fsdp` 则基于 DTensor 归约，可同时覆盖 TP 与 DP 两个维度。因此两条路径在复合场景下产生的 scale 不同，只有 `precompute` 路径保持跨 TP 维度的 single-device semantics。
- rowwise + TP 的具体结果取决于每次 GEMM 中 operand 的 `axiswise_dim` 与 DTensor Shard placement 的对应关系。需要注意的是，rowwise scale 在代码中作用于进入 `scaled_mm` 的 operand（通常是 weight.T），而非原始 weight，因此 Shard 维度在转置后会翻转。例如，column TP 下 weight 为 `Shard(1)`，weight.T 为 `Shard(0)`，此时 `amax(dim=-1)` reduce 的是 non-Shard 维度，本地完成；row TP 下则相反，需要 DTensor all-reduce。
- activation（input / grad_output）的 scale 不传入显式 `reduce_amax`，但其计算同样遵循 DTensor dispatch 语义。若 activation 为 `Replicate` placement（如 `Float8ColwiseParallel` 的 input），scale 在本地计算；若为 `Shard` placement（如 `Float8RowwiseParallel` 的 `Shard(-1)` input），`tensor_to_amax` 中的 `torch.max` 通过 DTensor dispatch 隐式触发跨 rank all-reduce。

stable float8 主线的实现链条如下：模块替换为 `Float8Linear`，前向与反向通过 `Float8TrainingTensor` 进入低精 GEMM，分布式路径中由 FSDP2 / DTensor 管理 `qdata + scale` 的通信与重建。`3.5` 起转入 MXFP8。

### 3.5 MXFP8 主线：模块替换与配置

以下小节转入 MXFP8 实现路径。该路径不继承前文 stable float8 的 `float8 all-gather` 语义，量化时机与通信策略需要单独分析。

MXFP8 的用户入口与 stable float8 不同。它不通过 `convert_to_float8_training()` 将 `nn.Linear` 替换为 `Float8Linear`，而是通过 `quantize_()` 将目标模块替换为 `MXLinear`：

```python
from torchao.quantization import quantize_
from torchao.prototype.mx_formats import MXLinearConfig

config = MXLinearConfig.from_recipe_name("mxfp8_cublas")
quantize_(model, config=config)
```

其底层注册链路如下：

```python
@register_quantize_module_handler(MXLinearConfig)
def _mx_linear_transform(module: torch.nn.Module, config: MXLinearConfig):
    return MXLinear.from_float(module, config=config)
```

该入口的关键点如下：

- `MXLinear.from_float()` 以原位方式将 `nn.Linear` 转为 `MXLinear`，不引入额外权重拷贝。
- 权重默认仍以高精度形式持久保存；MX 量化在前向和反向的 GEMM 入口按需执行。
- `MXLinearConfig` 的核心字段包括 `block_size=32`、`elem_dtype=torch.float8_e4m3fn`、`scale_calculation_mode` 以及 `kernel_preference`。
- 常见 recipe 中，`mxfp8_cublas` 与 `mxfp8_cublas_rceil` 对应 Blackwell / SM100+ 上的 cuBLAS MX GEMM 路径。

### 3.6 MXFP8 的前向与反向量化路径

MXFP8 训练同样围绕 `Linear` 的三次 GEMM 展开，但其量化粒度与反向实现路径比 stable float8 更具体。对应的核心逻辑位于 `mx_mm.apply(...)`：

```python
# input @ weight_t = output
input_mx_r_dim0 = MXTensor.to_mx(...)
weight_mx_dim0 = MXTensor.to_mx(...)
output = torch.mm(input_mx_r_dim0, weight_mx_dim0.t())

# grad_output @ weight = grad_input
grad_output_mx_dim0 = MXTensor.to_mx(...)
weight_mx_dim1 = _to_mxfp8_dim1_kernel_wrapper(...)  # 或转置后再 to_mx
grad_input = torch.mm(grad_output_mx_dim0, weight_mx_dim1.t())

# input_t @ grad_output = grad_weight
grad_output_mx_dim1 = _to_mxfp8_dim1_kernel_wrapper(...)
input_t_mx_dim0 = _to_mxfp8_dim1_kernel_wrapper(...).t()
grad_weight = torch.mm(grad_output_mx_dim1, input_t_mx_dim0)
```

该路径的关键特征如下：

- 前向中，`input` 与 `weight` 沿 dim0 量化为 MX 表示，再进入 GEMM。
- 反向中，`grad_output` 仍会先走 dim0 量化以计算 `grad_input`。
- 计算 `grad_input` 与 `grad_weight` 时，还需要额外的 dim1 colwise 量化路径。
- dim1 量化通常通过 `_to_mxfp8_dim1_kernel_wrapper()` 承担，是 MXFP8 backward 中的重要性能关键路径。

因此，MXFP8 的核心变化并不是“把参数永久存成 MX”，而是把三次 GEMM 的输入在算子入口动态转换为 `MXTensor`，再由 `torch.mm` 分发到相应的 MX kernel。

### 3.7 `MXTensor`、E8M0 scale 与 kernel 路径

`MXTensor` 是 MXFP8 训练中的核心表示。其内部保存的是：

- `qdata`：量化后的 `float8_e4m3fn` 数据。
- `scale`：`float8_e8m0fnu` 形式的 microscale。
- `block_size`：固定为 `32`。
- `_orig_dtype`：原始高精度 dtype，通常为 BF16。

`MXTensor.to_mx()` 的量化过程如下：

1. 按 `block_size=32` 将数据重排为 block 视图。
2. 计算每个 block 的 `max_abs`。
3. 根据 `ScaleCalculationMode` 将 `max_abs` 转为 `E8M0` scale。
4. 用该 scale 对 block 内数据归一化后 cast 到 `float8_e4m3fn`。

在当前实现中，`ScaleCalculationMode` 的工程含义如下：

- `FLOOR`：更接近 OCP MX 规范中的基础定义。
- `RCEIL`：更贴近当前 NVIDIA cuBLAS MX GEMM 路径，通常也是 TorchTitan 主推的 Blackwell recipe。

GEMM 执行路径分为两类：

- `KernelPreference.AUTO`：将 scale 转为 blocked / swizzled layout 后，调用 `torch._scaled_mm` 对接 cuBLAS MX GEMM。
- `KernelPreference.EMULATED`：先按 `MXTensor` 语义反量化回高精度，再执行普通 `mm`，用于仿真或非目标硬件。

### 3.8 stable float8 与 MXFP8 的通信时序对照

在分布式场景下，stable float8 tensorwise 与 MXFP8 的关键差异之一在于“量化发生在通信前还是通信后”。

| 阶段 | `float8 tensorwise` | `MXFP8` 主线 |
| --- | --- | --- |
| 本地初始状态 | 高精局部分片 / 本地 tensor | 高精局部分片 / 本地 tensor |
| 量化语义对齐 | `FSDP` 路径通常表现为显式 `all-reduce(MAX)` 或复用 `precomputed scale`；`TP / SP` 路径更常见的是在 `DTensor` 上计算 `max/amax`，当 scale 的定义域跨越 shard 边界时，由 reduction 语义跨 rank 对齐对应的 scale | 无额外 `amax` 同步主路径 |
| 通信前是否量化 | 是。先调用 `hp_tensor_to_float8_dynamic(...)` 将 tensor 转为 low precision 表示 | 否。保持高精度 tensor 进入 distributed communication |
| distributed communication | `float8 all-gather` / low precision `redistribute` | 高精度 `all-gather` / `redistribute`，通常为 BF16 或 `param_dtype` |
| 通信 payload | `qdata + scale` | 高精度 tensor |
| 通信后处理 | 重建 `Float8TrainingTensor`，再进入 `scaled_mm` | 进入 `MXLinear / mx_mm` 后，才调用 `MXTensor.to_mx(input / weight)` |
| 进入 kernel 前的最终表示 | `Float8TrainingTensor(qdata + scale)` | `MXTensor(qdata + E8M0 scale)` |
| 结果 | 低精表示直接参与 distributed communication | 量化发生在通信之后，distributed communication 本身不携带 MX 量化语义 |

在当前 TorchTitan + TorchAO 主线下，对应判断如下：

- stable float8 tensorwise 会先对齐 `amax/scale`，再将低精表示直接传入 distributed communication。各 recipe 与并行策略下 scale 通信的完整分析见 `3.4`。
- MXFP8 的 block-32 scale 始终在本地计算，不需要跨 rank 通信——前提是 block 不跨越 rank 边界。FSDP 下，当前主线通过高精度 all-gather 恢复完整权重后再本地量化，此条件天然满足；TP 下权重永久分片，需要 shape 约束保证 shard 边界与 block 边界对齐（见下方）。因此 MXFP8 不依赖 `precompute_float8_dynamic_scale_for_fsdp`。进一步地，若 FSDP shard 内 block 也不跨越 rank 边界（即 shard 的最后一维同样满足对 32 的整除约束），理论上可以先在本地执行 `to_mx()` 再 all-gather MX 表示，实现类似 stable float8 的低精通信；底层 `MXTensor` 的 all-gather 算子已实现（`mx_all_gather`），但训练路径中尚未集成对应的 FSDP hook。
- MXFP8 + TP 存在一个隐含的 shape 约束：`to_mx()` 内部将最后一维 reshape 为 `[K/32, 32]` 的 block 视图，该 reshape 经 DTensor dispatch 映射到 local shard。若 tensor 沿最后一维分片（如 `Shard(-1)`），则要求 `K/tp` 能被 32 整除；满足时 shard 边界与 block 边界天然对齐，不满足时 DTensor reshape 直接报错（当前不支持 padding）。对于典型 LLM 配置（hidden_dim = 4096/8192，TP = 2/4/8），该约束自然满足。

### 3.9 两条主线的组件关系

| 组件 | 主要职责 | 与低精路径的关系 |
| --- | --- | --- |
| `TorchTitan` | 训练入口、recipe 选择、配置装配与工程编排 | 决定启用 `float8` 还是 `MXFP8`，并将 converter 接入模型 |
| `TorchAO float8 / MXFP8` | 定义低精表示、量化逻辑与算子路径 | 将 `Linear/GEMM` 改造为低精执行路径 |
| `FSDP2 / DTensor` | 参数分片、unshard / all-gather、reduce-scatter、参数生命周期管理 | 决定分布式参数在通信前后处于何种表示 |
| `float8 all-gather` | 在 stable float8 tensorwise 路线中传输 `qdata + scale` | 使低精表示直接进入 distributed communication |
| `torch.compile` | 将 `amax/scale` 计算、cast、reshape 与 tensor subclass dispatch 收进编译图，并在可能时融合低精 GEMM 前后的 pointwise kernel | 决定低精 kernel 的收益能否转化为端到端收益 |
| `scaled_mm / rowwise / MX kernels` | 执行低精 GEMM | stable float8 主要对接 `scaled_mm`，MXFP8 对接 MX kernel / cuBLAS MX GEMM |
| `Optimizer step` | 更新高精度 master weight | 作为下一轮量化与通信的起点 |
| `precompute_float8_dynamic_scale_for_fsdp` | 为 stable float8 tensorwise FSDP 预计算下一轮 weight scale | 减少前向热路径中的 `amax` 归约开销；MXFP8 主线默认不依赖该步骤 |

上述关系可压缩为以下主链：

```text
TorchTitan
  -> TorchAO float8 / MXFP8 recipe
  -> FSDP2 / DTensor
  -> communication path
  -> torch.compile
  -> scaled_mm / rowwise / MX kernels
  -> forward / backward
  -> optimizer.step()
```

### 3.10 单次 iteration 时序

以下时序主要对应 `stable float8 + FSDP2` 路径，因为其中包含 `precompute_float8_dynamic_scale_for_fsdp` 与 `float8 all-gather`：

| 阶段 | 发生的动作 | 张量表示 |
| --- | --- | --- |
| `optimizer.step()` 后 | 更新 sharded master weight | 高精度分片参数 |
| 预处理阶段 | 调用 `precompute_float8_dynamic_scale_for_fsdp`，为下一轮 weight 计算并缓存 scale | 高精度 weight + 预计算 scale |
| FSDP2 pre-hook | 在 all-gather 前，将本地高精度 weight shard 转为 low precision 表示 | `qdata + scale` |
| distributed communication | 执行 `float8 all-gather` | 通信 payload 为 `qdata + scale` |
| forward GEMM | gathered weight 重建为 `Float8TrainingTensor`；activation 动态量化后执行 GEMM | `Float8TrainingTensor(input / weight)` |
| backward GEMM | `grad_output`、必要时的 `input` / `weight` 再次按配置量化后执行 dgrad / wgrad GEMM | `Float8TrainingTensor(grad_output / input / weight)` |
| backward 结束 | `reduce-scatter` 后回到 sharded 参数状态 | 高精度梯度 / 分片参数 |
| 下一轮开始前 | 等待下一次 `optimizer.step()` 完成后重复上述流程 | 高精度 master weight |

MXFP8 主线与上述时序的主要差异如下：

- distributed communication 默认保持高精度。
- `MXTensor.to_mx(...)` 发生在 `MXLinear / mx_mm` 的算子入口，而不是 `all-gather` 之前。

对应地，MXFP8 主线的简化时序如下：

| 阶段 | 发生的动作 | 张量表示 |
| --- | --- | --- |
| `optimizer.step()` 后 | 更新分片参数，参数仍以高精度形式持久保存 | 高精度分片参数 |
| distributed communication | 执行高精度 `all-gather` / `redistribute` | 高精度 tensor |
| forward / backward GEMM 前 | 在 `MXLinear / mx_mm` 入口调用 `MXTensor.to_mx(...)`，将输入张量转换为 MX 表示 | `MXTensor(qdata + E8M0 scale)` |
| kernel 执行后 | 执行 MX GEMM，输出回到高精度训练语义 | 高精度输出 / 梯度 |

## 4. 路线选择与实现判断

### 4.1 稳定性与可用性优先

适用路线为 `torchao.float8` 的 stable 路线：

- H100 / H200 上经验更成熟。
- TorchTitan 已提供直接集成。
- 与 FSDP2 / DTensor / `torch.compile` / float8 all-gather 的组合更完整。

### 4.2 吞吐优先

起始选择通常为 `tensorwise`：

- 通常是最快路径。
- 通信侧也更容易与 float8 all-gather 配合。
- 但对 outlier 更敏感。

### 4.3 loss parity / outlier 优先

适用路线通常为 `rowwise`，必要时使用 `rowwise_with_gw_hp`：

- 吞吐通常低于 `tensorwise`。
- 但数值稳定性通常更好。

### 4.4 Blackwell / B200 且可接受 prototype

评估对象可扩展到 MXFP8：

- 它更贴近 Blackwell 原生能力。
- 但当前 API 与 kernel 仍在快速演进。
- 端到端是否优于 stable float8，仍需结合 shape、compile 与通信路径联合评估。

### 4.5 常见误区

常见误解是：

- “启用 FP8 = 全模型所有算子都变成 FP8”。

根据当前官方资料，对应结论如下：

- 主要优化对象仍然是 `Linear/GEMM`。
- 其他算子，尤其 attention，是否低精度化要看具体版本和实现进展。

## 5. 最小用法与入口

### 5.1 TorchTitan stable float8 入口

`Pre-training with float8` 教程中的最小示例如下。

rowwise：

```bash
NGPU=8 CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml" ./run_train.sh --training.compile --model.converters="float8" --float8.recipe_name="rowwise"
```

tensorwise：

```bash
NGPU=8 CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml" ./run_train.sh --training.compile --model.converters="float8"
```

该入口对应以下职责分工：

- TorchTitan 负责训练主循环、并行策略、配置和日志。
- `--model.converters="float8"` 负责把 float8 converter 接到模型上。
- `--float8.recipe_name` 用于选择具体 recipe。

### 5.2 直接用 TorchAO 接入 stable float8

在不使用 TorchTitan 的情况下，可直接在模型代码中接入：

```python
import torch
import torch.nn as nn
from torchao.float8 import Float8LinearConfig, convert_to_float8_training

m = nn.Sequential(
    nn.Linear(8192, 4096, bias=False),
    nn.Linear(4096, 128, bias=False),
).bfloat16().cuda()

config = Float8LinearConfig.from_recipe_name("tensorwise")
convert_to_float8_training(m, config=config)
m = torch.compile(m)
```

### 5.3 直接用 TorchAO 接入 MXFP8

MXFP8 在 TorchAO 中的直接入口通常为 `quantize_()`：

```python
import torch
import torch.nn as nn
from torchao.quantization import quantize_
from torchao.prototype.mx_formats import MXLinearConfig

m = nn.Sequential(
    nn.Linear(8192, 4096, bias=False),
    nn.Linear(4096, 128, bias=False),
).bfloat16().cuda()

config = MXLinearConfig.from_recipe_name("mxfp8_cublas")
quantize_(m, config=config)
m = torch.compile(m)
```

这一入口与 stable float8 的主要差异在于：

- `quantize_()` 会将目标 `nn.Linear` 原位转换为 `MXLinear`。
- 权重通信默认仍保持高精度，MX 量化发生在 `mx_mm` 的前向和反向 GEMM 入口。
- 若目标是 Blackwell / B200 上的 cuBLAS MX GEMM，应优先选择 `mxfp8_cublas` 或 `mxfp8_cublas_rceil`。

## 6. 参考资料

1. [Supercharging Training using float8 and FSDP2](https://pytorch.org/blog/training-using-float8-fsdp2/)
2. [Pre-training with float8](https://docs.pytorch.org/ao/stable/eager_tutorials/pretraining.html)
3. [Quantized Training](https://docs.pytorch.org/ao/stable/workflows/training.html)
4. [Accelerating 2K scale pre-training up to 1.28x with TorchAO, MXFP8 and TorchTitan on Crusoe B200 Cluster](https://pytorch.org/blog/accelerating-2k-scale-pre-training-up-to-1-28x-with-torchao-mxfp8-and-torchtitan-on-crusoe-b200-cluster/)
5. [Float8LinearRecipeName API](https://docs.pytorch.org/ao/stable/api_reference/generated/torchao.float8.Float8LinearRecipeName.html)
6. [precompute_float8_dynamic_scale_for_fsdp API](https://docs.pytorch.org/ao/stable/generated/torchao.float8.precompute_float8_dynamic_scale_for_fsdp.html)
7. [torchao.float8.fsdp_utils source](https://docs.pytorch.org/ao/stable/_modules/torchao/float8/fsdp_utils.html)
8. [FSDP2 / fully_shard docs](https://docs.pytorch.org/docs/2.8/distributed.fsdp.fully_shard.html)
9. [TorchAO Quantization Overview](https://docs.pytorch.org/ao/stable/contributing/quantization_overview.html)
10. [TorchAO MX Formats Documentation](https://github.com/pytorch/ao/tree/main/torchao/prototype/mx_formats)

> 最后整理日期：2026-04-14
>
> 备注：本文主要基于 2024-11-25 到 2026-03-25 期间的官方博客、TorchAO 文档与 API/源码说明整理。MXFP8 相关结论默认建立在其仍处于快速演进阶段这一前提上；如果后续面向 Blackwell 生产环境，建议结合最新 release note、kernel benchmark 与 nightly 状态重新确认。
