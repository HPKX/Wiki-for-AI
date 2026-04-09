# Transformer Engine FP8 / MXFP8：收益、机制与 PyTorch 实现

本文按以下三个部分展开：

1. 收益及收益来源
2. TE FP8、MXFP8 核心思想和机制
3. TE + PyTorch 代码实际实现

总体上，TE 并不是将整个训练过程统一改为 FP8 dtype，而是在模块边界内维护一套低精度运行时：它统一管理 recipe、quantizer、quantized tensor、GEMM kernel、通信重叠以及 amax/scale 状态同步。

本文的范围与版本锚定如下：

- 范围：正文聚焦 TE 中的 `FP8` 与 `MXFP8` 训练路径。`NVFP4` 仅在个别位置作为格式或硬件演进的横向参照出现，不展开其数值格式、执行链与工程实现。
- 版本：源码路径与行为分析主要基于当前工作区中的 `TransformerEngine` 仓库，`build_tools/VERSION.txt` 为 `2.12.0.dev0`，对应 commit `69636a08171d162a223ae3a01e4b36902b64dba1`。TE 迭代较快，若目标环境版本不同，应以对应版本源码为准。

---

## 1. 收益及收益来源

### 1.1 核心目标：把最贵的 GEMM 主路径稳定推入低精度训练

训练里最贵的三类矩阵乘法通常是：

```text
forward:   y  = x @ w^T
backward:  dx = dy @ w
backward:  dw = x^T @ dy
```

> FP8/TE 的核心不是“把模型压成 8 bit 存储”，而是“把 forward、dgrad、wgrad 这三条最贵的 GEMM 主路径稳定地纳入低精度训练”。

这里的关键不是单一 `cast`，而是把量化、scale/amax 维护、量化缓存复用、kernel 调度以及必要的通信与同步组织成一条可训练的低精度执行链路。

从成本模型看，主收益主要来自两项：

- 主干 GEMM 进入低精度后带来的吞吐提升：在支持 FP8 的硬件上，Tensor Core 的 FP8 矩阵乘吞吐通常高于 BF16/FP16
- 参与 GEMM 的 activation、weight、gradient 位宽下降后带来的关键路径带宽压力下降：位宽减半意味着同等带宽下可搬运更多数据

对应的抵消项主要包括：

- 量化与相关数据搬运开销
- amax 记录与 scale 更新开销
- 量化缓存维护开销
- 通信与状态同步开销

因此，实际端到端收益并不等于理论 dtype 收益，而是“低精度 GEMM 吞吐与带宽收益”减去“量化、状态管理与通信开销”后的净收益。

### 1.2 附加收益的边界：关键路径带宽和存储压力可能下降，但取决于具体实现形态

低位宽运行时通常会带来关键路径带宽压力下降，并在部分场景下降低存储压力，但这一点需要结合 TE 的实际实现形态来理解：

- 只有参与 GEMM 的 activation、weight、gradient 才会以量化形式出现
- 模块内部常同时存在高精度语义、量化副本以及为 backward 或不同布局准备的缓存
- 只有显式使用 `quantized_model_init(enabled=True, ...)` 时，模块参数才会偏向“只持有量化副本”

因此，TE 可以降低关键路径上的带宽和部分存储压力，但实际显存收益仍取决于 recipe、模块类型、是否保留高精度参数、是否需要缓存行向/列向量化副本以及是否维护主权重；实际端到端收益还会受到量化、通信、状态同步等附加开销抵消。

### 1.3 工程价值：把低精度训练里的数值和分布式细节收敛成统一运行时

从工程视角看，TE 的价值不只是提供低精度格式本身，而是把低精度训练里的数值与分布式细节收敛成一套统一运行时。FP8 的难点主要在于：

- scale 怎么算
- amax 在什么时候记录
- forward 和 backward 用同一套还是不同一套 scale
- 分布式里 amax 是否同步、怎么同步
- backward 需要的 transpose/quantized tensor 如何缓存

TE 将这些问题统一收敛到：

```text
recipe + recipe state + quantizer + quantized tensor + global state manager
```

这意味着业务侧通常不需要自己维护：

- `amax_history`
- scale update 公式
- distributed `all_reduce(MAX)`
- quantized tensor 的保存和恢复逻辑

换言之，TE 的工程价值在于把低精度训练从“零散技巧”变成统一运行时，把数值稳定性细节从模块实现中抽离出来，让使用者主要关心“模块和 recipe”，而不是自己维护一套 FP8 状态机。

### 1.4 公开资料中的收益信息

截至当前参考资料范围，`Transformer Engine` 官方资料主要提供机制说明、可用性描述以及定性收益表述，尚未给出统一可直接引用的 `TE + PyTorch FP8/MXFP8` 端到端 benchmark。下面列出的收益数字主要来自 `PyTorch / TorchAO / TorchTitan` 生态公开资料，可作为 FP8/MXFP8 训练收益的旁证，而不应视为 `Transformer Engine` 官方 benchmark。

#### 数值格式信息

- `E4M3` 的可表示范围约为 `±448`
- 在本文讨论的 `TE + PyTorch` 路径里，常见实现是 `float8_e4m3fn`；它保留 `NaN`，但不提供 `infinity` 编码
- `E5M2` 的可表示范围约为 `±57,344`，并保留 `infinity / NaN`
- 这解释了为什么前向常偏向 `E4M3`，而反向/梯度更常偏向 `E5M2`：后者不仅动态范围更大，也更适合承接可能出现极值或溢出的梯度

#### 生态侧公开结果


| 场景                                                               | 公开结果                               | 含义                         |
| ---------------------------------------------------------------- | ---------------------------------- | -------------------------- |
| Crusoe B200 / 2K scale pre-training / TorchAO + TorchTitan MXFP8 | `1.22x–1.28x`                      | 说明 MXFP8 已经能转化为端到端训练收益     |
| Llama4 Scout / GB200 / TorchAO + TorchTitan / MoE                | `+30.2%` training speedup，`~1.3x` | 说明 MXFP8 在 MoE 训练里也能得到稳定加速 |


这些结果更适合作为“生态侧旁证”，用于说明 FP8/MXFP8 训练收益已经在公开系统中得到验证；在技术汇报中应显式标注来源，避免与 `Transformer Engine` 官方资料混写。

生态数据来源如下：

- `TorchAO + TorchTitan + MXFP8`，Crusoe B200，Llama3-70B：`1.22x - 1.28x` training acceleration vs BF16  
来源：[https://pytorch.org/blog/accelerating-2k-scale-pre-training-up-to-1-28x-with-torchao-mxfp8-and-torchtitan-on-crusoe-b200-cluster/](https://pytorch.org/blog/accelerating-2k-scale-pre-training-up-to-1-28x-with-torchao-mxfp8-and-torchtitan-on-crusoe-b200-cluster/)
- `TorchAO + TorchTitan + MXFP8`，Llama4 Scout，GB200，256 GPUs：`+30.2%` training speedup，约 `1.3x`；`BF16 5317 tokens/sec`，`MXFP8 6921 tokens/sec`  
来源：[https://pytorch.org/blog/mxfp8-training-for-moes-1-3x-training-speedup-vs-bf16-for-llama4-scout-on-gb200-cluster-using-torchao-and-torchtitan/](https://pytorch.org/blog/mxfp8-training-for-moes-1-3x-training-speedup-vs-bf16-for-llama4-scout-on-gb200-cluster-using-torchao-and-torchtitan/)
- `Float8 in PyTorch [1/x]`：float8 相对 16-bit 的 `2x` 吞吐/内存理论优势；`float8_experimental` 小规模训练 `up to 1.2x`；`LLaMA 7B / 1 GPU: 1.22x`；`LLaMA 13B / 8 GPUs: 1.20x`  
来源：[https://dev-discuss.pytorch.org/t/float8-in-pytorch-1-x/1815](https://dev-discuss.pytorch.org/t/float8-in-pytorch-1-x/1815)

### 1.5 这些收益成立的前提

这些前提可分为两类：一类决定 TE FP8/MXFP8 能否启用并正确运行，另一类决定这些低精度路径能否进一步转化为明显的端到端收益。

#### 能启用并正确运行

- 硬件支持：`is_fp8_available()` / `is_mxfp8_available()`
- 模块支持：收益集中在 `Linear`、`LayerNormLinear`、`LayerNormMLP`、attention 等已接入路径
- shape 约束：以 `Linear` 的 FP8 路径为例，公开文档说明参与 GEMM 的 `M/N/K` 相关维度需要满足对齐条件，常见要求是可被 `16` 整除；`MXFP8` 还有更严格对齐要求
- recipe / 调用方式约束：当 `recipe.reduce_amax=True` 时，同一模块不能在同一个 `autocast` 区域里被重复调用，因为 amax reduction 在退出 `autocast` 时统一执行，重复调用会覆盖待归约的 amax 状态

#### 能转化为明显收益

- 主干 GEMM 占比足够高：收益主要来自 forward、dgrad、wgrad 等主路径矩阵乘法，因此模型结构和训练热点需要让这些路径占据足够大的训练时间比例
- shape 足够规整且规模足够大：只有当 GEMM 本身足够“吃满”硬件时，低精度 Tensor Core 的吞吐优势才更容易转化为端到端收益
- 附加开销不过度抵消收益：量化、amax/scale 更新、缓存维护、通信和同步等额外运行时开销不能过大，否则理论收益会被明显吞掉

本文后续主要讨论运行机制与工程实现，不讨论训练收敛性、loss 曲线表现以及更细的数值稳定性边界。

---

## 2. TE FP8、MXFP8 核心思想和机制

### 2.1 TE 是模块级低精度运行时，而不是全局 dtype 切换

常见表述中容易出现一种误解：

> TE 就是把模型转成 FP8。

更准确的表述应为：

> TE 在模块边界内把高精度 tensor 转成低精度 kernel 可消费的表示，并维护这种表示在前向、反向、通信和状态更新中的生命周期。

这意味着：

- 不是所有 op 都会变成 FP8
- 不是所有 tensor 都长期以 FP8 形式存储
- loss、optimizer step、很多非 GEMM 算子仍保持正常的高精度语义

### 2.2 单层 `Linear` 的最小闭环

若以单层 `te.Linear` 配合 `DelayedScaling` 为例，一次训练迭代中的最小闭环可概括为：

```text
forward:
  x --量化--> x_fp8，记录 amax_x
  w --量化--> w_fp8，记录 amax_w
  y = GEMM(w_fp8, x_fp8)
  保存 backward 所需的量化缓存

forward 结束:
  汇总 amax_x / amax_w
  必要时做 all_reduce(MAX)
  更新下一轮 forward 要使用的 scale

backward:
  dy --量化--> dy_fp8，记录 amax_dy
  dx = GEMM(w_cached, dy_fp8)
  dw = GEMM(x_cached, dy_fp8)

backward 结束:
  汇总 amax_dy
  必要时做 all_reduce(MAX)
  更新下一轮 backward 要使用的 scale
```

这说明 TE 讨论的并不是某一次孤立的 `cast`，而是一次完整的“量化 -> GEMM -> 状态更新”闭环。

### 2.3 FP8 训练依赖的是“量化数据 + scale + kernel”

上述闭环可以进一步抽象为下面这个近似式：

```text
q = cast_to_fp8(x * scale)
```

kernel 实际消费的并不是“裸 FP8 dtype”，而是：

- 量化后的数据
- 对应的 scale / scale_inv
- 知道如何解释它们的 GEMM kernel

从数值表示的角度看，重点并不只是 `torch.float8_e4m3fn` / `torch.float8_e5m2` 这样的 dtype 名称，而是“量化数据 + scaling tensor + scaled matmul”这一整套表示与执行方式。

### 2.4 recipe 的作用：定义 scale 从哪里来

TE 中最重要的几类 recipe，本质上都在回答同一个问题：

> 当前这个 tensor 的 scale，应该怎么得到？

从 per-tensor 到 per-block，scaling 粒度逐步细化。粒度越细，每个 scale 覆盖的值域越窄，量化误差越小；但相应地，scale 存储开销更大，对齐约束也更严格。下面三种 recipe 分别代表这条粒度轴上的三个位置。

#### `DelayedScaling`

这是最经典的 per-tensor scaling 方案。

核心逻辑是：

- 本轮量化使用当前已有的 scale
- 本轮量化时记录新的 `amax`
- forward 或 backward 结束后再统一更新下一轮要用的 scale

优点是：

- 路径清晰
- 适合批量 `amax` reduction 和统一 scale update
- 工程上最成熟

#### `Float8CurrentScaling`

它仍然是 per-tensor scaling，但 scale 直接来自当前输入，而非历史窗口。

二者可以概括为：

- `DelayedScaling`：上一轮更新、下一轮使用
- `Float8CurrentScaling`：本轮即算、本轮即用

#### `MXFP8BlockScaling`

它不再采用“每个 tensor 一个 scale”的方式，而是 block scaling。

核心特征：

- 按 OCP MX 规范，1D microscaling 的 block size 固定为 `32`，因此每 `32` 个连续值共享一个缩放因子；这不是 TE 在 recipe 层暴露的可配置项
- 缩放因子是 `E8M0` 的 power-of-two 形式
- 量化方向和布局有关
- 一个 MXFP8 tensor 的 transpose 与原 tensor 不再数值等价

因此，TE 在 MXFP8 路径中通常需要分别维护 rowwise 和 columnwise 的量化副本。

与 delayed scaling 不同，`MXFP8` 的 scale 在量化时按 block 即时计算，因此不经过跨 rank 的 `amax all_reduce`。

这进一步约束了并行切分的边界。`MXFP8Quantizer` 要求最后一维与外层展平维都能被 `32` 整除；只有当 shard 边界与 block 边界对齐时，局部张量才能继续保持原生 MXFP8 表示。

当通信路径上的局部形状或数据形态不满足这一条件时，TE 的处理策略可分为三种情况：

- 未量化输入在 `all-gather`、FSDP `all-gather` 等通信路径上的局部形状不满足 block 对齐：先执行高精度通信，再对通信结果重新量化
- 输入已经是 `MXFP8Tensor`，但当前路径所需的 rowwise / columnwise usage 不匹配：先在本地反量化，再按所需 usage 重新量化后继续低精度通信
- `split`、`slice` 或 FSDP 分片本身破坏了 block 对齐：退回通用高精度张量语义，而不再跨 rank 共同维护同一个 block

### 2.5 TE 的核心抽象链

若将前面的最小闭环映射到运行时抽象，可得到下面这条链：

```text
recipe
  -> recipe state
  -> quantizer
  -> quantized tensor
  -> GEMM kernel / communication kernel
```

各层职责如下：

- `recipe`：定义缩放策略和格式
- `recipe state`：模块实际持有的运行时状态，例如 `scale`、`amax_history`
- `quantizer`：定义高精度 tensor 如何被量化
- `quantized tensor`：定义量化结果在 PyTorch 中如何表达、如何缓存 transpose/scale 元数据
- `kernel`：定义低精度 GEMM、cast、通信重叠和 scale update 的底层执行

因此，`DelayedScaling`、`Float8CurrentScaling`、`MXFP8BlockScaling` 虽然是不同的 recipe 选项，但它们都作用在同一条运行时主链上；某一层选择哪一种 recipe，就决定该层后续的量化方式、状态形态、tensor 表示以及 scale/amax 的更新流程。

### 2.6 用户侧最常见的入口：`te.autocast(...)`

对训练脚本而言，最常见、也是最重要的用户侧入口是 `te.autocast(...)`。

它定义的是一次低精度执行上下文，主要负责：

- 启用当前低精度 recipe
- 检查硬件和 recipe 支持
- 让模块在 forward 期间拿到统一的 quantizer / fp8 meta
- 在退出时触发 delayed scaling 的 forward 侧状态更新

它解决的是“这一次前向/反向如何运行”。

与之不同，`te.quantized_model_init(...)` 只在需要讨论“参数以什么形式初始化与持有”时才相关，不属于 FP8 执行主线。本文后续重点讨论 `autocast` 驱动的执行路径，不再单独展开 `quantized_model_init(...)`。

---

## 3. TE + PyTorch：`DelayedScaling` 主线，及 `Float8CurrentScaling`、`MXFP8` 差异

本节先按训练脚本、内部主链、运行前准备、forward、底层落点、backward、运行时收尾与并行路径的顺序展开 `FP8 / DelayedScaling` 主线，随后集中归纳 `Float8CurrentScaling` 与 `MXFP8` 相对该主线的差异。

在实现层面，可分为两层：

- `ops/basic/basic_linear.py`：对应 forward/backward/GEMM 主线的最小闭环
- `module/linear.py`、`module/layernorm_linear.py`：在最小闭环上再叠加 tensor parallel、userbuffers overlap、FSDP 等工程能力

其中，`3.1` 到 `3.10` 围绕 `FP8 / DelayedScaling` 主线展开；`3.11` 归纳 `Float8CurrentScaling` 相对该主线的关键差异；`3.12` 再归纳 `MXFP8` 相对该主线的关键差异。按展开层次看，`3.1` 到 `3.4` 主要位于训练脚本与 Python runtime 的桥接层；`3.5` 下沉到 C++ 绑定与 kernel 落点，偏源码实现细节；`3.6` 之后再回到 backward、状态更新与并行路径。

### 3.1 用户视角：训练脚本写什么，TE 接管什么

最小训练脚本可写为：

```python
layer = te.Linear(hidden_size, hidden_size).cuda()
recipe = DelayedScaling(...)

for x, target in dataloader:
    x = x.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with te.autocast(enabled=True, recipe=recipe):
        y = layer(x)
        loss = loss_fn(y, target)
    loss.backward()
    optimizer.step()
```

从训练脚本视角看，用户显式负责的部分主要是：

- 选择是否用 `te.Linear`、`te.LayerNormLinear` 等 TE 模块替换原模块
- 选择 recipe，并决定 `te.autocast(...)` 的作用范围
- 保持原有 loss、`backward()`、optimizer step、训练循环逻辑

TE 在模块内部接管的部分主要是：

- 构造 `recipe state`、quantizer 和量化 tensor
- 在 forward/backward 中对进入低精度路径的 activation、weight、gradient 执行量化
- 选择并发起对应的低精度 GEMM kernel
- 维护 `amax_history`、scale 及其同步更新
- 在需要时处理 tensor parallel 通信与通信重叠

从用户代码到 TE 内部实现之间的桥接可概括为：

```text
用户写：
  模块替换 + recipe 配置 + autocast 边界 + 常规训练循环

TE 接管：
  quantizer/state 创建
  -> 进入低精度路径的 activation/weight/grad 量化
  -> 低精度 GEMM 调度
  -> amax/scale 更新
  -> 可选通信与 overlap
```

训练脚本中新增的显式代码主要包括“替换模块 + 指定 recipe + 使用 `autocast`”；量化、状态更新、GEMM 调度和同步逻辑由 TE runtime 负责。

### 3.2 内部主链：`FP8 / DelayedScaling` 的执行骨架

从用户脚本进入 TE runtime 之后，内部主链可概括为：

```text
autocast 只做“状态切换”
  -> 模块把 recipe 具体化成 recipe state + quantizers
  -> quantizer 把 BF16/FP16 张量变成 QuantizedTensor
  -> general_gemm() 把量化张量、输出量化参数、通信重叠参数打包
  -> tex.generic_gemm() 发起底层 GEMM
  -> forward/backward 结束后再统一更新 amax/scale
```

`autocast()` 的作用不是“把后续所有算子自动 cast 成 FP8”，而是“为模块提供一套低精度运行时”。

### 3.3 运行前准备：全局状态与量化器初始化（`FP8 / DelayedScaling`）

进入内部主链后，执行首先进入运行前准备阶段：一部分由 `autocast()` 建立全局上下文，另一部分由模块把 recipe 具体化成运行时状态与量化器。

#### 3.3.1 `autocast()` 负责建立全局运行时状态

`transformer_engine/pytorch/quantization.py` 中的 `autocast()` 主要完成两项工作：

1. 进入时调用 `FP8GlobalStateManager.autocast_enter(...)`
2. 退出时调用 `FP8GlobalStateManager.autocast_exit(...)`

`autocast_enter(...)` 并不执行量化，而是将当前低精度上下文登记到全局状态：

- `FP8_ENABLED`
- `FP8_RECIPE`
- `FP8_DISTRIBUTED_GROUP`
- `AUTOCAST_DEPTH`
- `IS_FIRST_FP8_MODULE`

同时会检查 recipe 是否被当前硬件支持。至此，尚未有任何输入张量被 cast。

关键代码片段如下：

```python
@contextmanager
def autocast(
    enabled: bool = True,
    calibrating: bool = False,
    recipe: Optional["Recipe"] = None,
    amax_reduction_group: Optional["dist_group_type"] = None,
    _graph: bool = False,
) -> None:
    if enabled:
        check_recipe_support(recipe)

    fp8_state = FP8GlobalStateManager.get_autocast_state()

    FP8GlobalStateManager.autocast_enter(
        enabled=enabled,
        calibrating=calibrating,
        fp8_recipe=recipe,
        fp8_group=amax_reduction_group,
        _graph=_graph,
    )
    try:
        yield
    finally:
        FP8GlobalStateManager.set_autocast_state(fp8_state)
        FP8GlobalStateManager.autocast_exit(enabled, _graph=_graph)
```

#### 3.3.2 模块把 recipe 具体化为“状态 + 量化器”

将 recipe 转为可执行对象的核心位置，是 `transformer_engine/pytorch/ops/op.py` 中的 `BasicOperation.reset_recipe_state()`。

对 `BasicLinear` 来说，最关键的代码链是：

```text
reset_recipe_state(recipe)
  -> RecipeState.create(recipe, mode="forward"/"backward", num_quantizers=...)
  -> recipe_state.make_quantizers()
  -> self._fp8_metas[...] / self._quantizers[...]
  -> FP8GlobalStateManager.add_fp8_tensors_to_global_buffer(...)
```

对 `DelayedScaling`，这里会构造 `DelayedScalingRecipeState`，其持有的核心状态为：

```python
scale: Tensor[num_quantizers]
amax_history: Tensor[history_len, num_quantizers]
```

`make_quantizers()` 会把每个量化槽位变成一个 `Float8Quantizer`：

```python
Float8Quantizer(
    self.scale[i],
    self.amax_history[0][i].reshape((1,)),
    self.dtype,
)
```

这一过程说明了两个关键事实：

- `DelayedScaling` 并非只保存一个全局 scale，而是每个 quantizer 槽位各有自己的 `scale[i]`
- quantizer 持有的 `amax` 指针直接指向 `amax_history[0][i]`，因此量化 kernel 写入的就是模块自身的运行时状态

其关系可表示为：

```text
recipe = DelayedScaling(...)
    |
    v
RecipeState.create(...)
    |
    +--> scaling_fwd: scale_fwd + amax_history_fwd
    |
    +--> scaling_bwd: scale_bwd + amax_history_bwd
    |
    v
make_quantizers()
    |
    +--> input_quantizer       -> 指向 scale_fwd[0], amax_history_fwd[0][0]
    +--> weight_quantizer      -> 指向 scale_fwd[1], amax_history_fwd[0][1]
    +--> grad_output_quantizer -> 指向 scale_bwd[0], amax_history_bwd[0][0]
```

仅以“入口在 `autocast()`”概括该过程并不充分。真正决定量化行为的是模块实例内部这套 `recipe state + quantizer`。

关键代码片段如下：

```python
recipe_state = RecipeState.create(
    recipe,
    mode=mode,
    num_quantizers=num_quantizers,
)
fp8_meta_key = FP8GlobalStateManager.get_meta_tensor_key(
    forward=(mode == "forward"),
)
self._fp8_metas[mode] = {
    fp8_meta_key: recipe_state,
    "recipe": recipe,
    "fp8_group": fp8_group,
}

self._quantizers[mode] = recipe_state.make_quantizers()
```

`DelayedScalingRecipeState` 的实现形式如下：

```python
self.scale = torch.ones(num_quantizers, dtype=torch.float32, device=device)
self.amax_history = torch.zeros(
    recipe.amax_history_len,
    num_quantizers,
    dtype=torch.float32,
    device=device,
)

return [
    Float8Quantizer(self.scale[i], self.amax_history[0][i].reshape((1,)), self.dtype)
    for i in range(self.num_quantizers)
]
```

### 3.4 计算主线：forward（`FP8 / DelayedScaling`）

完成运行前准备后，执行链进入 forward 主线。对应实现位于 `ops/basic/basic_linear.py` 中的 `op_forward()` 和 `_functional_forward()`。

#### 3.4.1 forward 前配置 quantizer usage

`BasicLinear.pre_fuser_forward()` 会根据当前 forward/backward 的需求设置：

- `input_quantizer.set_usage(rowwise=True, columnwise=weight_requires_grad)`
- `weight_quantizer.set_usage(rowwise=True, columnwise=False)`
- `grad_output_quantizer.set_usage(rowwise=True, columnwise=weight_requires_grad)`

关键不在于“rowwise/columnwise 只是布局标签”，而在于：

- forward GEMM 需要哪种视图
- backward dgrad / wgrad 是否还要复用这份量化数据
- 是否需要提前保留 transpose cache

这里 `weight_quantizer` 在 forward 阶段仅设置 `rowwise` usage，并不意味着训练过程中始终不需要 `columnwise` weight。若 backward 的 dgrad 路径需要按列消费 weight，TE 会在后续缓存复用或 `update_usage(...)` 阶段补充生成相应视图，详见 `3.6`。

这些 usage 设置决定了后续 GEMM 与 backward 是否能直接复用同一份量化缓存，而不是简单的布局标签切换。

对普通 `FP8` 而言，`columnwise` 在部分设备路径上并非仅对应逻辑转置，而是由 `Float8TensorStorage` 额外维护的一份转置后 `FP8` buffer。该类通过 `_data` 与 `_transpose` 分别保存 rowwise 与 columnwise 所需的数据；只有设备支持 `non-TN FP8 GEMM` 时，`columnwise` 才可以直接通过 GEMM 的 layout 参数处理，无需额外的 transpose cache；否则仍需要 `Float8TensorStorage` 维护独立的转置 buffer。

#### 3.4.2 `_functional_forward()` 的执行链路

`_functional_forward()` 的逻辑可压缩为下图：

```text
x_local
  -> 如果是 column TP + sequence parallel:
       gather_along_first_dim(..., quantizer=input_quantizer)
     否则:
       input_quantizer(x_local)
  -> x

w
  -> 如果还不是 quantized tensor:
       weight_quantizer(w)
  -> w_q

wait(x_async)
  -> general_gemm(w_q, x, layout="TN", quantization_params=output_quantizer)
  -> y

如果是 row TP:
  -> reduce_scatter_along_first_dim(y) 或 all_reduce(y)

为 backward 做缓存:
  -> x_local.update_usage(...)
  -> w_q.update_usage(...)
```

这一执行链还包含三个实现细节。

第一，forward 的 GEMM 在代码中写成：

```python
y, *_ = general_gemm(
    w,
    x,
    out_dtype=dtype,
    quantization_params=output_quantizer,
    ...
)
```

`general_gemm()` 默认 `layout="TN"`，对应 `y = x @ w^T`。这表明 Python 层并不手工转置矩阵，而是把布局语义交给底层 GEMM。

第二，输入通信和量化可以合并处理。例如在 column parallel + sequence parallel 场景下，`gather_along_first_dim(..., quantizer=input_quantizer)` 表明 TE 不一定遵循“先 gather 高精度，再单独量化”的流程，而是允许通信路径直接产出量化张量。

第三，forward 结束后不会直接丢弃量化张量，而是通过 `update_usage(...)` 为 backward 预留合适的数据形态。例如：

- `w.update_usage(rowwise_usage=False, columnwise_usage=True)`
- `x_local.update_usage(rowwise_usage=False, columnwise_usage=True)`

这对应于“forward 量化一次，backward 尽量复用”的具体代码实现。

关键代码片段如下：

```python
x_local = input
x = None
x_async = None
with_x_all_gather = tensor_parallel_mode == "column" and sequence_parallel
if with_quantized_compute:
    if input_quantizer is None:
        raise ValueError("Missing quantizer for input tensor")
    input_quantizer.set_usage(rowwise=True, columnwise=weight_requires_grad)
    if with_x_all_gather:
        input_quantizer.set_usage(columnwise=False)
        x, x_async = gather_along_first_dim(
            x_local,
            tensor_parallel_group,
            async_op=True,
            quantizer=input_quantizer,
        )
    else:
        if not is_quantized_tensor(x_local):
            x_local = input_quantizer(x_local)
        x = x_local
```

forward GEMM 的调用位置如下：

```python
y, *_ = general_gemm(
    w,
    x,
    out_dtype=dtype,
    quantization_params=output_quantizer,
    alpha=alpha,
    beta=beta,
    accumulate=accumulate_into_out,
    out=y,
    bias=bias,
    use_split_accumulator=_2X_ACC_FPROP,
)
```

其中，`_2X_ACC_FPROP`、`_2X_ACC_DGRAD`、`_2X_ACC_WGRAD` 是 TE 内部用于控制是否启用 split accumulator 的开关；本文仅标注其在三次 GEMM 中的出现位置，不展开其数值与性能权衡。

### 3.5 底层落点：`tex.quantize()` 和 `tex.generic_gemm()`（`FP8 / DelayedScaling`）

在 forward 主线中，量化和 GEMM 已经分别落到 `tex.quantize()` 与 `general_gemm()`。继续追踪实际执行位置，可进一步下沉到 C++ 绑定和底层 kernel。这一部分属于实现细节。

#### 3.5.1 量化的 C++ 绑定与 kernel 落点

在 `transformer_engine/pytorch/tensor/float8_tensor.py` 里，`Float8Quantizer.quantize_impl()` 只有一行：

```python
return tex.quantize(tensor, self)
```

对应的 C++ 绑定在 `pytorch/csrc/extensions/cast.cpp`：

```text
tex.quantize(...)
  -> convert_quantizer(...)
  -> quantizer_cpp->create_tensor(...) / convert_and_update_tensor(...)
  -> quantizer_cpp->quantize(input_cpp, output_cpp, noop_flag_cpp)
```

继续下沉会进入 TE 公共层的 `nvte_quantize_v2(...)`。对 delayed scaling 来说，底层 kernel 主要完成两件事：

1. 用当前 `scale` 把输入量化成 FP8
2. 用 `atomicMaxFloat` 把本次 `amax` 写回 quantizer 指向的 `amax_history[0]`

更准确的表述是：“量化器在 Python 中发起量化调用，amax 的写回由量化 kernel 在生成量化结果时同步完成”。

对应的 Python 侧入口代码如下：

```python
def update_quantized(
    self,
    src: torch.Tensor,
    dst: QuantizedTensor,
    *,
    noop_flag: Optional[torch.Tensor] = None,
) -> QuantizedTensor:
    if not src.is_contiguous():
        src = src.contiguous()

    tex.quantize(src, self, dst, noop_flag)
    dst._fp8_dtype = self.dtype
    return dst

def quantize_impl(self, tensor: torch.Tensor) -> QuantizedTensor:
    return tex.quantize(tensor, self)
```

对应的 C++ 绑定是：

```cpp
py::object quantize(const at::Tensor &tensor, py::handle quantizer, const py::object &output,
                    std::optional<at::Tensor> noop_flag) {
  auto quantizer_cpp = convert_quantizer(quantizer);
  auto input_contiguous = tensor.contiguous();
  auto input_cpp = makeTransformerEngineTensor(input_contiguous);

  TensorWrapper output_cpp;
  py::object output_py;
  if (output.is_none()) {
    const auto shape = get_tensor_shape(input_cpp);
    const auto fake_dtype = input_cpp.dtype();
    std::tie(output_cpp, output_py) = quantizer_cpp->create_tensor(shape, fake_dtype);
  } else {
    std::tie(output_cpp, output_py) = quantizer_cpp->convert_and_update_tensor(output);
  }

  quantizer_cpp->quantize(input_cpp, output_cpp, noop_flag_cpp);
  return output_py;
}
```

#### 3.5.2 GEMM 的完整调用接口

`transformer_engine/pytorch/cpp_extensions/gemm.py` 中的 `general_gemm()` 会把下列信息一起整理好：

- `A`、`B`
- `layout`
- `out_dtype`
- `quantization_params`
- `alpha` / `beta` / `accumulate`
- `ub` / `ub_type` / `extra_output` / `bulk_overlap`

最后统一下传到：

```python
out, bias_grad, gelu_input, extra_output = tex.generic_gemm(*args, **kwargs)
```

因此，TE 的 GEMM 不是“先量化，后续再处理通信和输出”，而是从接口层面就把这些运行时信息绑定成一次调用。

关键代码片段如下：

```python
def general_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,
    quantization_params: Optional[Quantizer] = None,
    ...
    ub: Union[tex.CommOverlap, tex.CommOverlapP2P] = None,
    ub_type: tex.CommOverlapType = None,
    extra_output: Optional[torch.Tensor] = None,
    bulk_overlap: bool = False,
) -> Iterable[Optional[torch.Tensor]]:
    ...
    out, bias_grad, gelu_input, extra_output = tex.generic_gemm(*args, **kwargs)
    return out, bias_grad, gelu_input, extra_output
```

### 3.6 计算主线：backward（`FP8 / DelayedScaling`）

完成 forward 及其底层调用链之后，执行路径进入 backward。`BasicLinear.op_backward()` 最终调用 `_functional_backward()`。其整体执行过程可先概括如下：

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│  loss.backward()                                                            │
│      │                                                                      │
│      ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  BasicLinear.op_backward()                                          │   │
│  │  ───────────────────────────────────────────────────────────────────│   │
│  │                                                                     │   │
│  │  1. 量化 grad_output (同时记录 amax)                                 │   │
│  │     dy_fp8 = grad_output_quantizer(grad_output)                     │   │
│  │           └─► backward amax_history[0] 被更新                       │   │
│  │                                                                     │   │
│  │  2. dgrad GEMM: 计算 grad_input                                     │   │
│  │     dx = general_gemm(W^T, dy_fp8, layout="NN")                     │   │
│  │     ├─ W^T 使用 forward 时缓存的 columnwise FP8 数据                │   │
│  │     └─ 输出为高精度 (BF16/FP16)                                     │   │
│  │                                                                     │   │
│  │  3. wgrad GEMM: 计算 grad_weight                                    │   │
│  │     dw = general_gemm(x^T, dy_fp8, layout="NT")                     │   │
│  │     ├─ x^T 使用 forward 时缓存的 columnwise FP8 数据                │   │
│  │     └─ 输出为高精度 (BF16/FP16)                                     │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  backward 结束后:                                                           │
│      │                                                                      │
│      └─► reduce_and_update_fp8_tensors(forward=False)                       │
│              (更新 backward 的 scale)                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

结合 `BasicLinear` 的源码实现，可将该执行路径进一步细化为：

```text
grad_output = dy_local
  -> 如果需要低精度:
       grad_output_quantizer.set_usage(
           rowwise=input_requires_grad,
           columnwise=weight_requires_grad,
       )
       若 row TP + sequence parallel:
           gather_along_first_dim(..., quantizer=grad_output_quantizer)
       否则:
           grad_output_quantizer(dy_local)
  -> dy

如果需要 dgrad:
  w -> 准备成适合列方向消费的量化视图
  wait(dy_async)
  dx = general_gemm(w, dy, layout="NN", grad=True)
  如果是 column TP:
      reduce_scatter(dx) 或 all_reduce(dx)

如果需要 wgrad:
  x -> 准备成适合列方向消费的量化视图
  wait(x_async, dy_async)
  dw = general_gemm(x, dy, layout="NT", grad=True)
```

其数据流可表示为：

```text
Backward

grad_output(dy)
  -> 量化 / all-gather
  -> dy_q

                weight_q(columnwise)
                       |
                       v
dy_q --------------> general_gemm(layout="NN") ---> dx
                       |
                       +--> 若 column TP: RS / all-reduce

input_q(columnwise)
       |
       v
dy_q -> general_gemm(layout="NT") -------------> dw
```

关键不在 `NN` 和 `NT` 这两个字面布局本身，而在于 backward 持续复用 forward 已准备好的量化缓存和 usage 信息。否则每条分支都需要重新量化，收益会被显著削弱。

关键代码片段如下：

```python
grad_output_quantizer.set_usage(
    rowwise=input_requires_grad,
    columnwise=weight_requires_grad,
)
...
dx, *_ = general_gemm(
    w,
    dy,
    out_dtype=dtype,
    quantization_params=grad_input_quantizer,
    alpha=grad_input_alpha,
    beta=grad_input_beta,
    accumulate=accumulate_into_grad_input,
    layout="NN",
    out=dx,
    use_split_accumulator=_2X_ACC_DGRAD,
    grad=True,
)
...
dw, *_ = general_gemm(
    x,
    dy,
    out_dtype=dw_dtype,
    alpha=grad_weight_alpha,
    beta=grad_weight_beta,
    accumulate=accumulate_into_grad_weight,
    layout="NT",
    out=dw,
    use_split_accumulator=_2X_ACC_WGRAD,
    grad=True,
)
```

### 3.7 运行时收尾：scale 更新（`FP8 / DelayedScaling`）

对 delayed scaling 而言，forward 和 backward 并不是一次迭代的全部；在两条计算主线之后，还存在一段运行时收尾逻辑。量化阶段只负责记录本轮 amax，真正的 scale 更新发生在 forward/backward 结束后。

#### 3.7.1 模块先将自身状态挂到全局缓冲区

`BasicOperation.reset_recipe_state()` 的最后会调用：

```python
FP8GlobalStateManager.add_fp8_tensors_to_global_buffer(self._fp8_metas[mode])
```

对 delayed scaling，这会把每个模块的：

- `amax_history[0]`
- `amax_history`
- `scale`

挂到 `global_amax_buffer`、`global_amax_history_buffer`、`global_scale_buffer` 中。

因此，后续执行的是“批量更新所有模块”，而不是每层 forward 结束后立即分别更新。

#### 3.7.2 forward 和 backward 的更新时机

- forward 侧：`FP8GlobalStateManager.autocast_exit()` 在 `AUTOCAST_DEPTH == 0` 时调用 `reduce_and_update_fp8_tensors(forward=True)`
- backward 侧：`ops/fuser.py` 与经典模块路径 `module/linear.py`、`module/layernorm_linear.py` 都是在各自 backward 尾部判断是否触发 `reduce_and_update_fp8_tensors(forward=False)`；其中 `is_first_fp8_module()` 在同一 `autocast` 上下文内只会返回一次，因此其实际语义是在整个 backward 过程结束时统一更新，而不是在每个模块的 backward 尾部分别更新一次

两者的更新时机分别为：

- forward scale 更新发生在 `autocast` 退出点
- backward scale 更新位于整次 backward 的收尾点，即所有参与该次 FP8 backward 的模块完成后再统一更新

#### 3.7.3 `reduce_and_update_fp8_tensors()` 的实际工作

这条函数链可表示为：

```text
各模块量化时写入各自 amax_history[0]
    |
    v
global_amax_buffer[key] = [mod1.amax0, mod2.amax0, ...]
    |
    v
contiguous_amax = torch.cat(amax_buffer)
    |
    +--> 如果 reduce_amax=True 且多卡:
    |      all_reduce(MAX)
    |
    v
tex.fused_amax_and_scale_update_after_reduction(...)
    |
    +--> 滚动 amax_history
    +--> 根据 recipe.amax_compute_algo 选 amax
    +--> 按 fp8_max / amax / 2^margin 更新 scale
    +--> 把新的 scale 留给下一轮量化使用
```

其关键点在于：

> delayed scaling 不是“本轮先算 amax，再立刻用它量化本轮”；而是“本轮量化使用旧 scale，本轮只记录新 amax，统一更新后供下一轮使用”。

关键代码片段如下：

```python
for buffer_key, amax_buffer in cls.global_amax_buffer.items():
    fwd_update, autocast_key = cls.split_key_in_buffer(buffer_key)
    if fwd_update != forward:
        continue
    if len(amax_buffer) == 0:
        continue

    recipe, group = cls.autocast_arguments[autocast_key]
    contiguous_amax = torch.cat(amax_buffer)

    if (
        recipe.reduce_amax
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size(group=group) > 1
    ):
        cls.reduce_tensor_across_group_op_max(contiguous_amax, group)

    tex.fused_amax_and_scale_update_after_reduction(
        contiguous_amax,
        cls.global_amax_history_buffer[buffer_key],
        cls.global_scale_buffer[buffer_key],
        recipe.amax_compute_algo,
        get_fp8_te_dtype(recipe, forward),
        recipe.margin,
    )
```

### 3.8 并行路径：通信在 GEMM 主线中的位置（`FP8 / DelayedScaling`）

在 `BasicLinear` 中，通信围绕 GEMM 排布，而不是独立附着在外层：

- forward 中，column parallel + sequence parallel 在 GEMM 前做 `gather_along_first_dim(...)`
- forward 中，row parallel 在 GEMM 后做 `reduce_scatter_along_first_dim(...)` 或 `all_reduce(...)`
- backward 中，row parallel + sequence parallel 在 dgrad / wgrad 前处理 `grad_output`
- backward 中，column parallel 在 dgrad 后处理 `grad_input`

在 `module/linear.py` 和 `module/base.py` 中，TE 还支持将 tensor-parallel 通信与 GEMM 做 overlap 的工程优化，例如 userbuffers。本文只保留这一能力在执行链中的位置，不展开其专用缓冲区、stream 调度与实现细节。

```python
general_gemm(
    ...,
    ub=ub_obj,
    ub_type=ub_type,
    extra_output=reduce_scatter_out,
    bulk_overlap=...,
)
```

这说明 overlap 不是 Python 外层额外包裹的一层逻辑，而是从 `general_gemm()` 这一层开始作为参数正式下传给 `tex.generic_gemm()`。

可归纳为：

- `BasicLinear` 展示的是最小可理解闭环
- `module/linear.py` 展示的是把 tensor parallel、userbuffers、FSDP、权重缓存等工程逻辑叠加到同一个闭环上
- 两条路径共享的核心都不是“入口 API”，而是同一条 `quantizer -> quantized tensor -> general_gemm -> tex.generic_gemm -> amax/scale update` 主链

### 3.9 并行路径：amax 同步（`FP8 / DelayedScaling`）

并行路径中的 `amax` 同步不由“是否发生通信”决定，而由“后续某个低精度 GEMM 或 attention kernel 是否会将来自不同 rank 的量化分片作为同一个逻辑 tensor 共同消费”决定。

因此，本节只讨论进入低精度 kernel 前仍处于量化语义中的 tensor，不讨论 GEMM 后的 `all_reduce`、`reduce_scatter` 等高精度输出通信。

在这一前提下，`DelayedScaling` 与 `Float8CurrentScaling` 的差异主要体现在归约发生的时间点，而不改变“为什么需要归约”这一判断标准。

| 并行策略 / 场景 | 同步哪个 tensor 的 `amax` | 同步组 | 触发时机 | 原因 |
| --- | --- | --- | --- | --- |
| `column parallel + sequence parallel` forward | 输入 activation，对应 `input_quantizer` / `GEMM1_INPUT` | `tp_group` | `Float8CurrentScaling` 在量化前执行 `all_reduce(MAX)`；`DelayedScaling` 在退出 `autocast` 时统一 `reduce_and_update` | 各 rank 持有的是同一个逻辑输入 `X` 的 sequence shard；这些分片会在 GEMM 前 `all-gather` 成完整 `X`，并作为同一个低精度 GEMM 的输入，因此必须共享同一量化标尺 |
| `row parallel + sequence parallel` backward | 输出梯度，对应 `grad_output_quantizer` / `GRAD_OUTPUT1` | `tp_group` | `Float8CurrentScaling` 在量化前执行 `all_reduce(MAX)`；`DelayedScaling` 在 backward 尾部统一 `reduce_and_update` | 各 rank 持有的是同一个逻辑 `dy` 的 sequence shard；这些分片会在 dgrad / wgrad 前 `all-gather` 成完整 `dy`，并作为低精度 GEMM 的输入，因此必须共享同一量化标尺 |
| `context parallel` attention | attention 相关 quantizer 的 `amax`，例如 `Q/K/V/O` 或 DPA quantizer | `DelayedScaling` 使用外层配置的 `fp8_group`，通常应覆盖 `TP+CP`；`Float8CurrentScaling` 使用 `cp_group`，`a2a+p2p` 时为 `cp_group[0]` | `Float8CurrentScaling` 在量化前即时归约；`DelayedScaling` 在 attention 的 FP8 状态更新阶段统一归约 | attention 内部会交换远端 `KV` 或重排 `QKV`；这些分片若作为同一逻辑上下文进入同一个低精度 attention kernel，就必须共享同一量化标尺 |
| `DTensor / FSDP2` 分片参数 | 权重 quantizer 的 `amax` | `DeviceMesh` 的 shard group | 参数量化时 | 分片参数仍属于同一个逻辑权重张量；若各 rank 需要共享统一量化语义，则必须在分片组内同步 `amax` |

注：在当前仓库对应的 TE 版本中，`a2a+p2p` 使用分层 CP 组，即 `cp_group = [a2a_group, p2p_group]`。其中 `Float8CurrentScaling` 的 `amax` 同步显式绑定到 `a2a_group`，也就是 `cp_group[0]`，而不是 `p2p_group`。因此，这里记录的是当前版本的具体实现行为，而不是跨版本保持不变的接口约定。不同 TE 版本里 CP 的通信组织与 `amax` 同步策略曾有调整，迁移时应以目标版本源码为准。

对于 `tensor parallel + sequence parallel` 组合，需要额外同步 `amax` 的核心场景为：

- `column parallel + sequence parallel` forward 的 `input`
- `row parallel + sequence parallel` backward 的 `grad_output`

其余围绕 GEMM 后高精度 tensor 展开的通信不属于本节讨论范围，因为这些路径中的 tensor 已不再携带低精度量化标尺。

### 3.10 总体执行链（`FP8 / DelayedScaling`）

将前面的用户视角、内部主链、计算主线、底层落点、运行时收尾和并行路径合并后，可得到下面这条总体执行链：

```text
with te.autocast(recipe=DelayedScaling(...)):
    |
    +--> FP8GlobalStateManager.autocast_enter()
    |
    +--> BasicLinear.reset_recipe_state()
    |       -> DelayedScalingRecipeState(scale, amax_history)
    |       -> Float8Quantizer
    |       -> add_fp8_tensors_to_global_buffer()
    |
    +--> forward
            -> input / weight quantize
            -> 可选 all-gather
            -> general_gemm(..., layout="TN")
            -> tex.generic_gemm(...)
            -> 可选 reduce-scatter / all-reduce
            -> 保存量化缓存给 backward

exit autocast
    -> reduce_and_update_fp8_tensors(forward=True)

loss.backward()
    |
    +--> grad_output quantize / communication
    +--> dgrad: general_gemm(..., layout="NN", grad=True)
    +--> wgrad: general_gemm(..., layout="NT", grad=True)
    +--> reduce_and_update_fp8_tensors(forward=False)
```

### 3.11 `Float8CurrentScaling` 相对 `FP8 / DelayedScaling` 主线的差异

前述 `3.1` 到 `3.10` 以 `FP8 / DelayedScaling` 为主线，用于先固定 `recipe state -> quantizer -> quantized tensor -> GEMM -> amax/scale update` 这条执行骨架。`Float8CurrentScaling` 在模块组织、`Float8Tensor` 表示以及 GEMM 主线上的大体框架与其保持一致，但在 `scale/amax` 的来源、同步时机与状态形态上存在关键差异。

#### 3.11.1 运行前准备差异

对 `Float8CurrentScaling`，`reset_recipe_state()` 会构造 `Float8CurrentScalingRecipeState`。它与 `DelayedScalingRecipeState` 的关键区别是：前者不维护 `scale` 与 `amax_history` 这类持久状态，而只记录当前 `mode` 对应的量化 dtype 与 device，并直接生成 `Float8CurrentScalingQuantizer`：

```python
class Float8CurrentScalingRecipeState(RecipeState):
    def __init__(...):
        self.recipe = recipe
        self.mode = mode
        self.num_quantizers = num_quantizers
        self.dtype = get_fp8_te_dtype(recipe, mode == "forward")
        self.device = device

    def make_quantizers(self) -> list:
        return [Float8CurrentScalingQuantizer(self.dtype, device=self.device) for i in range(self.num_quantizers)]
```

因此，`Float8CurrentScaling` 在运行前准备阶段的重点不再是“把 delayed scaling 的状态张量挂到模块上”，而是“确定当前 forward/backward 需要的 quantizer 集合、dtype 以及 workspace buffer 所在设备”。

#### 3.11.2 量化与状态更新差异

对 `Float8CurrentScaling`，`tex.quantize()` 沿用与 `DelayedScaling` 相同的 Python 入口，但底层语义不同：不再”读取已有 `scale` 并将新 `amax` 写回 `amax_history[0]`”，而是直接从当前输入计算 `amax`，由该 `amax` 得到 `scale` 并完成 cast。量化器定义本身已体现了这一点：

```python
class Float8CurrentScalingQuantizer(Quantizer):
    """...compute amax directly from current input..."""

    def quantize_impl(self, tensor: torch.Tensor) -> QuantizedTensor:
        return tex.quantize(tensor, self)
```

对应的底层步骤可概括为：

```text
tex.quantize(src, Float8CurrentScalingQuantizer)
  -> 从当前输入计算 amax
  -> 若当前并行路径要求共享量化标尺，则先执行 amax all_reduce(MAX)（见 3.9）
  -> 由当前 amax 计算 scale
  -> cast 为 Float8Tensor
```

其中”哪些并行场景需要共享量化标尺”的判断标准与 `DelayedScaling` 相同，具体场景见 `3.9` 节并行同步表。区别在于：`DelayedScaling` 的 `amax` 归约发生在 `autocast_exit()` 或 backward 收尾时的 `reduce_and_update_fp8_tensors(...)` 中；而 `Float8CurrentScaling` 的归约在量化时即时完成。因此，`reduce_and_update_fp8_tensors(...)` 在 current scaling 路径上即使被调用也是 `noop`，因为 delayed scaling 专用的全局 `amax` buffer 为空。

量化结果类型仍然是 `Float8Tensor`，forward/backward 的缓存复用与 GEMM 主链总体不变。与 `DelayedScaling` 相比，核心差异集中在 `amax/scale` 的来源与更新时机，而不是张量表示或 GEMM 调度本身。

#### 3.11.3 总体执行链

其总体执行链可简化为：

```text
with te.autocast(recipe=Float8CurrentScaling(...)):
    |
    +--> FP8GlobalStateManager.autocast_enter()
    |
    +--> BasicLinear.reset_recipe_state()
    |       -> Float8CurrentScalingRecipeState(mode, dtype, device)
    |       -> Float8CurrentScalingQuantizer
    |       -> add_fp8_tensors_to_global_buffer(...)  # noop
    |
    +--> forward
            -> input / weight 从当前输入计算 amax
            -> 若需要，先做 amax all_reduce(MAX)
            -> 计算当前 scale 并量化为 Float8Tensor
            -> general_gemm(...)
            -> 保存可复用的量化缓存给 backward

exit autocast
    -> reduce_and_update_fp8_tensors(forward=True)  # noop

loss.backward()
    |
    +--> grad_output 从当前输入计算 amax / scale 后量化为 Float8Tensor
    +--> dgrad / wgrad
    +--> reduce_and_update_fp8_tensors(...)  # noop
```

### 3.12 `MXFP8` 相对 `FP8 / DelayedScaling` 主线的差异

`MXFP8` 与 `Float8CurrentScaling` 不同，它在量化状态、张量表示、backward 数据消费以及并行通信上都与 `FP8 / DelayedScaling` 主线存在更大差异。

#### 3.12.1 运行前准备差异

对 `MXFP8BlockScaling`，`reset_recipe_state()` 会构造 `MXFP8BlockScalingRecipeState`。它与 `DelayedScalingRecipeState` 的关键区别是：前者不维护 `scale` 与 `amax_history` 这类持久状态，而只根据 `mode` 选择当前量化 dtype，并直接生成 `MXFP8Quantizer`：

```python
class MXFP8BlockScalingRecipeState(RecipeState):
    def __init__(...):
        self.recipe = recipe
        self.mode = mode
        self.num_quantizers = num_quantizers
        self.dtype = get_fp8_te_dtype(recipe, mode == "forward")

    def make_quantizers(self) -> list:
        return [MXFP8Quantizer(self.dtype) for i in range(self.num_quantizers)]
```

因此，`MXFP8` 在运行前准备阶段的重点不再是“把 delayed scaling 的状态张量挂到模块上”，而是“确定当前 forward/backward 需要的 quantizer 集合与 dtype”。虽然 `reset_recipe_state()` 仍统一调用 `add_fp8_tensors_to_global_buffer(...)`，但该入口对非 delayed recipe 是 `noop`。

#### 3.12.2 量化与张量表示差异

对 `MXFP8`，`tex.quantize()` 虽然沿用与 `Float8Quantizer` 相同的 Python 入口，但底层语义已经从“读取当前 `scale` 并写回 `amax_history[0]`”切换为“按 block 即时计算 scale 并生成 `MXFP8Tensor`”。量化器定义本身已经体现了这一点：

```python
class MXFP8Quantizer(Quantizer):
    """...dividing them into groups of 32 elements..."""

    def quantize_impl(self, tensor: torch.Tensor) -> QuantizedTensor:
        return tex.quantize(tensor, self)
```

结合 `MXFP8` 的 usage 约束，可将其底层量化结果概括为：

```text
tex.quantize(src, MXFP8Quantizer)
  -> 每 32 个元素计算局部 scale
  -> 生成 rowwise_data + rowwise_scale_inv
  -> 若当前 usage 需要，再生成 columnwise_data + columnwise_scale_inv
  -> 返回 MXFP8Tensor
```

因此，`tex.quantize()` 在 `MXFP8` 路径中既负责块级量化，也决定后续 GEMM 与通信可直接复用的 rowwise / columnwise 表示；这里不再对应 delayed scaling 那种 `amax_history` 写回与统一更新流程。

#### 3.12.3 backward 数据消费差异

对 `MXFP8`，usage 管理比普通 `FP8` 更关键。forward 通常会保留 `x_local` 的 columnwise 版本给 wgrad 复用；backward 中新量化得到的 `grad_output` 则往往同时需要 rowwise 与 columnwise 两种消费方向。对应关系可概括为：

- dgrad：`W columnwise` + `dY rowwise`
- wgrad：`X columnwise` + `dY columnwise`

这意味着 `MXFP8` backward 并非“量化出一份 `dy_fp8` 后直接复用于两个 GEMM”。尤其在 `row parallel + sequence parallel` 场景下，若 dgrad 路径已经先 `all-gather` 了 rowwise `grad_output`，这份结果不能直接转换成 wgrad 所需的 columnwise 版本；必要时需要为 wgrad 单独准备 columnwise `grad_output`。

#### 3.12.4 并行路径中的通信差异

在 `MXFP8BlockScaling` 下，并行路径中的额外关注点不再是 `amax` 同步，而是通信对象是否仍保持为 `MXFP8Tensor` 的块级表示。`MXFP8` 采用局部即时 scale，因此不存在 delayed scaling 那种跨 rank 的 `all_reduce(MAX)`；在满足 block 对齐与 usage 约束时，低精通信主要体现为 `all-gather` 直接传输 `data + scale_inv`，不满足这些条件时再退回高精路径或本地重新量化。

| 并行策略 / 场景 | 引入的通信 | 通信对象 | 是否属于 MXFP8 低精通信 | 说明 |
| --- | --- | --- | --- | --- |
| `column parallel + sequence parallel` forward | GEMM 前 `all-gather` | `input` 的 rowwise `data + scale_inv` | 是 | `gather_along_first_dim(..., quantizer=...)` 可直接拼出完整 `MXFP8Tensor`；forward GEMM 消费 rowwise 输入 |
| `row parallel + sequence parallel` backward，dgrad 路径 | GEMM 前 `all-gather` | `grad_output` 的 rowwise `data + scale_inv` | 是 | dgrad GEMM 使用 `dy rowwise` |
| `row parallel + sequence parallel` backward，wgrad 路径 | 额外一次 `all-gather`，或单独准备 columnwise 版本 | `grad_output` 的 columnwise `data + scale_inv` | 是 | wgrad GEMM 使用 `dy columnwise`；rowwise 结果不能直接复用为 columnwise |
| `row parallel` forward / `column parallel` backward dgrad | GEMM 后 `reduce-scatter` 或 `all-reduce` | `y` 或 `dx` | 否 | 通信对象已经是 `BF16/FP16` 输出，不再携带 MXFP8 标尺 |
| `context parallel` attention | `p2p` / `all-gather` / `a2a` / `a2a+p2p` | `KV/QKV` 或 attention 中间量 | 否 | 当前 `DotProductAttention` 的 FP8 attention 路径只支持 delayed/current per-tensor scaling，`MXFP8` 不在该低精 attention 主线中 |
| `DTensor / FSDP2` 分片参数 | 参数 `all-gather` | 分片权重的 rowwise / columnwise `data + scale_inv` | 是 | `reshard_after_forward=True` 时前向只 gather rowwise，反向只 gather columnwise；若前向后的参数不立即重分片，则前向可能同时 gather 两者 |
| `data parallel / DDP` | 梯度 `all-reduce` | 参数梯度 | 否 | 梯度同步仍在 `BF16/FP32` 下进行 |

因此，`MXFP8` 的低精通信主要集中在 `all-gather` 类路径，而不是 GEMM 后的高精度归约。通信对象也不是单独的 FP8 payload，而是与 GEMM 方向绑定的块级表示：`rowwise_data + rowwise_scale_inv` 或 `columnwise_data + columnwise_scale_inv`。

#### 3.12.5 总体执行链

```text
with te.autocast(recipe=MXFP8BlockScaling(...)):
    |
    +--> FP8GlobalStateManager.autocast_enter()
    |
    +--> BasicLinear.reset_recipe_state()
    |       -> MXFP8BlockScalingRecipeState(mode, dtype)
    |       -> MXFP8Quantizer
    |       -> add_fp8_tensors_to_global_buffer(...)  # noop
    |
    +--> forward
            -> input / weight 量化为 MXFP8Tensor
            -> 按 usage 准备 rowwise / columnwise 表示
            -> 可选低精度 all-gather
            -> general_gemm(...)
            -> 保存可复用的量化缓存给 backward

exit autocast
    -> reduce_and_update_fp8_tensors(forward=True)  # noop

loss.backward()
    |
    +--> grad_output 量化为 MXFP8Tensor
    +--> dgrad: 消费 W columnwise + dY rowwise
    +--> wgrad: 消费 X columnwise + dY columnwise
    +--> reduce_and_update_fp8_tensors(forward=False)  # noop
```

第 3 节可概括为：

> TE + PyTorch 的实现核心在于：模块将 recipe 具体化为运行时状态与 quantizer，并在此基础上将量化、GEMM、通信以及量化状态维护组织成一条闭环执行链。

---

## 4. 参考资料

- Using FP8 and FP4 with Transformer Engine  
[https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
- Per-Tensor and Per-Block Scaling Strategies for Effective FP8 Training  
[https://developer.nvidia.com/blog/per-tensor-and-per-block-scaling-strategies-for-effective-fp8-training/](https://developer.nvidia.com/blog/per-tensor-and-per-block-scaling-strategies-for-effective-fp8-training/)
- Floating-Point 8: An Introduction to Efficient, Lower-Precision AI Training  
[https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/)
- Transformer Engine PyTorch API  
[https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html)
- Transformer Engine 2.0 Release Notes  
[https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.0/release-notes/index.html](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.0/release-notes/index.html)
- Float8 in PyTorch [1/x]  
[https://dev-discuss.pytorch.org/t/float8-in-pytorch-1-x/1815](https://dev-discuss.pytorch.org/t/float8-in-pytorch-1-x/1815)
- Accelerating 2K scale pre-training up to 1.28x with TorchAO, MXFP8 and TorchTitan on Crusoe B200 Cluster  
[https://pytorch.org/blog/accelerating-2k-scale-pre-training-up-to-1-28x-with-torchao-mxfp8-and-torchtitan-on-crusoe-b200-cluster/](https://pytorch.org/blog/accelerating-2k-scale-pre-training-up-to-1-28x-with-torchao-mxfp8-and-torchtitan-on-crusoe-b200-cluster/)
- MXFP8 Training for MoEs: 1.3x training speedup vs BF16 for Llama4 Scout on GB200 cluster using TorchAO and TorchTitan  
[https://pytorch.org/blog/mxfp8-training-for-moes-1-3x-training-speedup-vs-bf16-for-llama4-scout-on-gb200-cluster-using-torchao-and-torchtitan/](https://pytorch.org/blog/mxfp8-training-for-moes-1-3x-training-speedup-vs-bf16-for-llama4-scout-on-gb200-cluster-using-torchao-and-torchtitan/)
