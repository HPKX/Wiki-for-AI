# MXFP8 rowwise / columnwise 源码梳理：input 与 weight 生命周期

## 1. 同一个逻辑 Tensor 的 rowwise / columnwise 为什么不等价

先固定讨论对象：这里说的是同一个逻辑矩阵 `X[M, K]`。`X.T` 是另一个逻辑 Tensor，形状、operand 语义和 scale 布局都已经变了，不能和“同一个 `X` 的 columnwise 表示”混在一起。

MXFP8 不是只把元素 dtype 换成 FP8。它的表示由两部分组成：

```text
MXFP8Tensor = FP8 data + E8M0 scale_inv
```

其中 scale 不是全局的，而是沿某个方向按 32 个连续元素一组计算。TE 的 recipe 文档直接定义了这个粒度：

```text
Each group of 32 consecutive values is scaled together using their own scaling factor.
```

证据：`../TransformerEngine/transformer_engine/common/recipe/__init__.py:269-272`

因此，rowwise / columnwise 的差异不是 stride 解释差异，而是 **scale 分组方向不同**。TE 在 GEMM 入口处也明确写了这一点：

```cpp
// Row-wise and column-wise data are scaled along different
// dimensions (with matrix interpreted in row-major order).
```

证据：`../TransformerEngine/transformer_engine/common/gemm/cublaslt_gemm.cu:175-178`

对同一个 `X[M, K]`，两种分组可以写成：

```text
rowwise(X):
  s_row[i, j_blk]
  覆盖 X[i, j_blk*32 : j_blk*32+32]

columnwise(X):
  s_col[i_blk, j]
  覆盖 X[i_blk*32 : i_blk*32+32, j]
```

所以同一个元素 `X[i, j]` 在两种表示里通常使用不同 scale：

```text
rowwise_data[i, j]     = quantize(X[i, j], s_row[i, j//32])
columnwise_data[i, j]  = quantize(X[i, j], s_col[i//32, j])
```

把这个差异落实到 scale 的形状上会更直观。对一个逻辑矩阵：

```text
X[M, K]
```

rowwise 是“每一行沿 K 方向切 block”：

```text
每行有 K / 32 个 block
共有 M 行
=> rowwise scale 逻辑形状是 [M, K / 32]
```

columnwise 是“每一列沿 M 方向切 block”：

```text
每列有 M / 32 个 block
共有 K 列
=> columnwise scale 逻辑形状是 [M / 32, K]
```

汇总成表：

| usage | scale 逻辑形状 | 含义 |
| --- | --- | --- |
| rowwise | `[M, K / 32]` | 每一行沿 K 方向，每 32 个连续元素一个 scale |
| columnwise | `[M / 32, K]` | 每一列沿 M 方向，每 32 个连续元素一个 scale |

两者的 scale 数量都是 `M * K / 32`，但索引含义不同：`s_row[i, j_blk]` 是“第 i 行的第 j_blk 个横向 block”，`s_col[i_blk, j]` 是“第 j 列的第 i_blk 个纵向 block”。所以它们不是同一个 scale buffer 换个 shape 解释，而是由不同 block 集合算出来的两套 scale。

再说一个容易误解的地方：既然 rowwise 和 columnwise 只是方向不同，那是不是把 `rowwise(X)` 的 `data` 和 `scale_inv` 都转置 / 重排，就得到了 `columnwise(X)`？

不是。这样得到的不是同一个逻辑 `X` 的 columnwise 量化，而是另一个逻辑 Tensor `X.T` 的 columnwise 量化。

下标关系如下：

```text
rowwise(X):
  q_row[i, j]      = quantize(X[i, j], s_row[i, j//32])
  s_row[i, j_blk]  覆盖 X[i, j_blk*32 : j_blk*32+32]

columnwise(X.T):
  q_col_T[j, i]      = quantize(X.T[j, i], s_col_T[j//32, i])
                    = quantize(X[i, j],   s_col_T[j//32, i])
  s_col_T[j_blk, i]  覆盖 X.T[j_blk*32 : j_blk*32+32, i]
                    = 覆盖 X[i, j_blk*32 : j_blk*32+32]

因此：
  q_col_T[j, i]      对应 q_row[i, j]
  s_col_T[j_blk, i]  对应 s_row[i, j_blk]
```

所以这只能说明：

```text
transpose_data_and_scale(quantize_rowwise(X))
  对应 quantize_columnwise(X.T)
```

它不能推出：

```text
quantize_rowwise(X)
  等价于 quantize_columnwise(X)
```

原因是 `quantize_columnwise(X)` 仍然要求同一个逻辑 `X[M, K]` 的 columnwise buffer，即 `columnwise_data + columnwise_scale_inv`；而对 `rowwise(X)` 的 `data/scale_inv` 做转置 / 重排，逻辑矩阵已经变成了 `X.T[K, M]`。

这也是 TE 里不能把 row-scaled MXFP8 临时当成 column-scaled MXFP8 的原因。源码在通信重叠路径里直接写明：

```python
# we can't convert row-scaled MXFP8 to column-scaled
```

证据：`../TransformerEngine/transformer_engine/pytorch/module/linear.py:787-792`

本节结论：

```text
同一个逻辑 X:
  quantize_rowwise(X) != quantize_columnwise(X)

物化转置后的 X.T:
  transpose_data_and_scale(quantize_rowwise(X))
    对应 quantize_columnwise(X.T)
```

## 2. 底层 GEMM 如何选择 rowwise / columnwise

先讲最容易反直觉的地方：`general_gemm(A_arg, B_arg, layout)` 的参数顺序，不等于数学公式里的左矩阵、右矩阵顺序。

从 TE 这层封装看，更应该把它理解成：

```text
general_gemm(A_arg, B_arg, layout)
  => op(B_arg) @ op(A_arg)
```

也就是说：

```text
B_arg / op(B_arg)  对应数学左矩阵
A_arg / op(A_arg)  对应数学右矩阵
```

源码的 output shape 能看出这个关系：输出维度先来自 B 侧外维，再来自 A 侧外维。

```cpp
NVTE_CHECK((transa ? A1 : A0) == (transb ? B0 : B1), ...);
...
// Construct output dims
if (transb) {
  ret.emplace_back(B1);
} else {
  // Unflatten B0
  ...
}
if (transa) {
  ret.emplace_back(A0);
} else {
  ret.emplace_back(A1);
}
```

证据：`../TransformerEngine/transformer_engine/pytorch/csrc/extensions/gemm.cpp:51-70`

然后再看 `layout`。`layout` 的第一个字符控制 `A_arg` 是否转置，第二个字符控制 `B_arg` 是否转置：

```python
layout: str = "TN"
...
transa = layout[0] == "T"
transb = layout[1] == "T"
```

证据：`../TransformerEngine/transformer_engine/pytorch/cpp_extensions/gemm.py:92-116`

因此一个 TE GEMM 调用要分两步理解：

```text
1. 先由 layout 得到 op(A_arg)、op(B_arg)
2. 再把数学公式写成 op(B_arg) @ op(A_arg)
```

从数学矩阵乘法看，rowwise / columnwise 的直觉规则是：

```text
数学乘法 L @ R:
  L 是左矩阵，沿 reduction 维按行消费 -> L rowwise
  R 是右矩阵，沿 reduction 维按列消费 -> R columnwise
```

把这条规则套回 TE 的 `A_arg / B_arg / layout`，就得到：

```text
layout="NN":
  op(A_arg) = A_arg
  op(B_arg) = B_arg
  math = B_arg @ A_arg

  B_arg 是数学左矩阵 -> B_arg rowwise
  A_arg 是数学右矩阵 -> A_arg columnwise

layout="TN":
  op(A_arg) = A_arg.T
  op(B_arg) = B_arg
  math = B_arg @ A_arg.T

  B_arg 是数学左矩阵 -> B_arg rowwise
  A_arg.T 是数学右矩阵 -> A_arg.T columnwise
  对应回原始 A_arg 存储，就是 A_arg rowwise

layout="NT":
  op(A_arg) = A_arg
  op(B_arg) = B_arg.T
  math = B_arg.T @ A_arg

  B_arg.T 是数学左矩阵 -> B_arg.T rowwise
  对应回原始 B_arg 存储，就是 B_arg columnwise
  A_arg 是数学右矩阵 -> A_arg columnwise
```

所以 `layout` 和 MXFP8 usage 的关系可以压缩成这个表：

| layout | 数学公式 | `A_arg` 需要 | `B_arg` 需要 |
| --- | --- | --- | --- |
| `NN` | `B_arg @ A_arg` | columnwise | rowwise |
| `TN` | `B_arg @ A_arg.T` | rowwise | rowwise |
| `NT` | `B_arg.T @ A_arg` | columnwise | columnwise |

这个表看起来和直觉并不冲突。真正容易让人绕进去的是：源码里的 `A_arg` 其实是数学右矩阵，源码里的 `B_arg` 才是数学左矩阵。

底层 GEMM 对 MXFP8 的 data / scale_inv 选择也正是这个规则。

### A_arg 选择规则

`A_arg` 是数学右矩阵那一侧。源码里：

```cpp
if (is_A_transposed) {
  NVTE_CHECK(A.has_data(), "Input A is missing row-wise usage");
} else {
  NVTE_CHECK(A.has_columnwise_data(), "Input A is missing column-wise usage");
}
ret.A = is_A_transposed ? A.data.dptr : A.columnwise_data.dptr;
ret.A_scale_inv = is_A_transposed ? A.scale_inv.dptr : A.columnwise_scale_inv.dptr;
```

证据：`../TransformerEngine/transformer_engine/common/gemm/cublaslt_gemm.cu:180-188`

对应为：

```text
A_arg transposed     -> A_arg rowwise
A_arg non-transposed -> A_arg columnwise
```

解释：当 `A_arg` 被转置参与计算时，数学右矩阵是 `A_arg.T`；`A_arg.T` 的 columnwise 对应原始 `A_arg` 的 rowwise。

### B_arg 选择规则

`B_arg` 是数学左矩阵那一侧。源码里：

```cpp
if (is_B_transposed) {
  NVTE_CHECK(B.has_columnwise_data(), "Input B is missing column-wise usage");
} else {
  NVTE_CHECK(B.has_data(), "Input B is missing row-wise usage");
}
ret.B = is_B_transposed ? B.columnwise_data.dptr : B.data.dptr;
ret.B_scale_inv = is_B_transposed ? B.columnwise_scale_inv.dptr : B.scale_inv.dptr;
```

证据：`../TransformerEngine/transformer_engine/common/gemm/cublaslt_gemm.cu:264-273`

对应为：

```text
B_arg transposed     -> B_arg columnwise
B_arg non-transposed -> B_arg rowwise
```

解释：当 `B_arg` 被转置参与计算时，数学左矩阵是 `B_arg.T`；`B_arg.T` 的 rowwise 对应原始 `B_arg` 的 columnwise。

最后补一个判断物化转置 Tensor 的规则。假设数学上要算：

```text
L @ R.T
```

如果不物化 `R.T`，TE 调用可以写成：

```text
general_gemm(R, L, layout="TN")
  => math = L @ R.T
  => R.T 是数学右矩阵，需要 columnwise
  => 对应原始 R 的 rowwise
```

如果先物化：

```text
C = R.T
L @ C
```

那 `C` 是一个新的逻辑 Tensor，作为数学右矩阵参与 `L @ C`，需要的是：

```text
C columnwise
```

这个 `C columnwise` 可以由原始 `R rowwise` 的 data / scale 转置重排得到，但它已经属于物化后的新逻辑 Tensor `C = R.T`，不是同一个 `R` 的 columnwise。

这条规则是后面推导 input / weight 生命周期的基础。

## 3. 核心结论

`Linear` 的三条 GEMM 主路径可以直接套用上面的 A/B 选择规则：

| 阶段 | TE 调用 | 数学含义 | MXFP8 表示需求 |
| --- | --- | --- | --- |
| forward | `general_gemm(weightmat, inputmat_total)`，默认 `layout="TN"` | `input @ weight.T` | 数学左矩阵 `input rowwise`；数学右矩阵 `weight.T columnwise`，对应原始 `weight rowwise` |
| dgrad | `general_gemm(weight_fp8, grad_output, layout="NN")` | `grad_output @ weight` | 数学左矩阵 `grad_output rowwise`；数学右矩阵 `weight columnwise` |
| wgrad | `general_gemm(inputmat_total, grad_output, layout="NT")` | `grad_output.T @ input` | 数学左矩阵 `grad_output.T rowwise`，对应原始 `grad_output columnwise`；数学右矩阵 `input columnwise` |

forward 调用证据：

```python
gemm_out, *_, reduce_scatter_out = general_gemm(
    weightmat,
    inputmat_total,
    ...
)
```

证据：`../TransformerEngine/transformer_engine/pytorch/module/linear.py:319-334`

dgrad 调用证据：

```python
weight_fp8.update_usage(columnwise_usage=True)
...
general_gemm(
    weight_fp8,
    grad_output,
    layout="NN",
    grad=True,
    ...
)
```

证据：`../TransformerEngine/transformer_engine/pytorch/module/linear.py:688-727`

wgrad 调用证据：

```python
wgrad_gemm_kwargs = {
    ...
    "layout": "NT",
    ...
}
...
dw, db, *_ = general_gemm(x, dy, **wgrad_gemm_kwargs)
```

证据：`../TransformerEngine/transformer_engine/pytorch/module/linear.py:850-884`

所以简化记忆是：

```text
forward:
  W rowwise + X rowwise

dgrad:
  W columnwise + dY rowwise

wgrad:
  X columnwise + dY columnwise
```

## 4. MXFP8 Tensor 组织形式

MXFP8 Tensor 在 Python storage 层把 rowwise 和 columnwise 拆成四个字段：

```text
_rowwise_data
_columnwise_data
_rowwise_scale_inv
_columnwise_scale_inv
```

证据：`../TransformerEngine/transformer_engine/pytorch/tensor/storage/mxfp8_tensor_storage.py:60-87`

`MXFP8Quantizer.make_empty()` 根据 usage 分别分配 rowwise 和 columnwise buffer：

```python
if self.rowwise_usage:
    data = torch.empty(...)
    scale_inv = torch.empty(...)

if self.columnwise_usage:
    columnwise_data = torch.empty(...)
    columnwise_scale_inv = torch.empty(...)
```

证据：`../TransformerEngine/transformer_engine/pytorch/tensor/mxfp8_tensor.py:120-146`

`update_usage()` 的语义也很关键：它可以检查某个方向是否存在，也可以把不需要的方向置空；但它不会把 rowwise 重新生成成 columnwise，或者反过来。

```python
if rowwise_usage:
    if self._rowwise_data is None:
        raise RuntimeError(...)
else:
    self._rowwise_data = None
    self._rowwise_scale_inv = None

if columnwise_usage:
    if self._columnwise_data is None:
        raise RuntimeError(...)
else:
    self._columnwise_data = None
    self._columnwise_scale_inv = None
```

证据：`../TransformerEngine/transformer_engine/pytorch/tensor/storage/mxfp8_tensor_storage.py:213-257`

源码注释也写明了限制：

```python
For MXFP8, columnwise scaled output is only produced by x2
scaling kernels, so this function only disables usages.
```

证据：`../TransformerEngine/transformer_engine/pytorch/tensor/storage/mxfp8_tensor_storage.py:218-220`

这意味着：

```text
如果 tensor 只有 rowwise，后续请求 columnwise 会报错
如果 tensor 同时有 rowwise + columnwise，可以通过 update_usage 丢弃其中一份
```

## 5. input 与 weight 生命周期

这一节只回答一个问题：同一个 Tensor 的 rowwise / columnwise 表示，在 forward 和 backward 之间到底怎么保留。

判断顺序是：

```text
1. forward GEMM 需要哪种 usage
2. backward GEMM 需要哪种 usage
3. save-for-backward 保存的是哪一个对象
4. 源码有没有主动丢弃某个 usage
```

按这个顺序看，input 和 weight 的策略不一样：

```text
input:
  forward 需要 rowwise
  backward wgrad 需要 columnwise
  forward 结束后源码会主动丢弃 rowwise

weight:
  forward 需要 rowwise
  backward dgrad 需要 columnwise
  forward 结束后保存 weightmat，源码没有主动丢弃 rowwise
```

### 5.1 input：forward 后可以裁掉 rowwise

input 是 activation 侧缓存，主要用于当前 forward / backward。它的 rowwise 是 forward GEMM 用的；它的 columnwise 是 backward wgrad 用的。二者用途不重叠，所以 TE 在 forward 结束后会尽量只把 backward 需要的 columnwise 保存下来。

forward 量化 input 时，无 input all-gather 路径会先声明 forward 需要 rowwise；如果 backward 需要 input，并且没有选择保存原始 input，则同时准备 columnwise：

```python
input_quantizer.set_usage(
    rowwise=True,
    columnwise=backward_needs_input and not save_original_input,
)
inputmat = input_quantizer(inputmat)
```

证据：`../TransformerEngine/transformer_engine/pytorch/module/linear.py:226-237`

input all-gather 路径也遵循同一个逻辑：本地 input 可以按 backward 需求准备 columnwise；但 gather 输出只服务 forward GEMM，因此只要求 rowwise：

```python
input_quantizer.set_usage(rowwise=True, columnwise=backward_needs_input)
...
quantizer = input_quantizer
quantizer.set_usage(rowwise=True, columnwise=False)
```

证据：`../TransformerEngine/transformer_engine/pytorch/module/linear.py:184-212`

forward GEMM 消费完 input rowwise 之后，如果保存给 backward 的是量化 input，TE 会显式丢弃 rowwise，只保留 columnwise：

```python
# Discard row-wise data since it is not needed in backward pass
inputmat.update_usage(rowwise_usage=False, columnwise_usage=True)
```

证据：`../TransformerEngine/transformer_engine/pytorch/module/linear.py:385-399`

后续保存到 backward 上下文的是这个已经更新 usage 的 `inputmat`：

```python
if backward_needs_input:
    saved_inputmat = inputmat
```

证据：`../TransformerEngine/transformer_engine/pytorch/module/linear.py:401-404`

backward 做 wgrad 前，TE 只要求 input 具备 columnwise。如果 forward 保存的是 quantized input，就检查已有 columnwise；如果保存的是原始 input，就在 backward 里按 `rowwise=False, columnwise=True` 重新量化：

```python
if isinstance(inputmat_total, QuantizedTensorStorage):
    inputmat_total.update_usage(columnwise_usage=True)
else:
    ctx.input_quantizer.set_usage(rowwise=False, columnwise=True)
    inputmat_total = ctx.input_quantizer(inputmat_total)
```

证据：`../TransformerEngine/transformer_engine/pytorch/module/linear.py:777-782`

所以 input 的生命周期可以概括为：

```text
forward quantize:
  input rowwise
  如果 backward 需要 wgrad，可能同时准备 input columnwise

forward GEMM:
  消费 input rowwise

forward save-for-backward:
  若保存的是 quantized input，丢弃 input rowwise
  保存 backward 需要的 input columnwise

backward wgrad:
  消费 input columnwise
```

结论：input 可能在 forward 阶段短暂同时持有 rowwise + columnwise，但源码有明确的裁剪点。反向 wgrad 不需要 input rowwise，所以它不会作为 saved activation 长时间保留。

### 5.2 weight：保存的是参数 workspace，rowwise 没有被裁掉

weight 的情况不同。`weightmat` 不是一次性的 activation，而是由参数 `weight` 量化得到的参数侧 workspace。forward 需要它的 rowwise，backward dgrad 又需要它的 columnwise；forward 结束保存到 backward 上下文的正是这个 `weightmat`。

forward 准备 weight 时，rowwise 始终打开；如果当前处于训练并且 input 需要 dgrad，columnwise 也会打开：

```python
columnwise_usage = is_grad_enabled and inp.requires_grad
...
weight_quantizer.set_usage(rowwise=True, columnwise=columnwise_usage)
```

证据：`../TransformerEngine/transformer_engine/pytorch/module/linear.py:252-264`

随后用这个 quantizer 生成 `weightmat`。这里的 `update_usage(rowwise_usage=True)` 只是确保 forward 所需的 rowwise 可用，不会关闭 columnwise：

```python
weightmat = module.get_weight_workspace(
    tensor=weight,
    quantizer=weight_quantizer,
    cache_name=(None if is_first_microbatch is None else "weight"),
    ...
)
weightmat.update_usage(rowwise_usage=True)
```

证据：`../TransformerEngine/transformer_engine/pytorch/module/linear.py:267-278`

forward 结束后，保存给 backward 的就是 `weightmat`：

```python
tensors_to_save, tensor_objects = prepare_for_saving(
    saved_inputmat,
    weightmat,
    weight,
    bias,
)
```

证据：`../TransformerEngine/transformer_engine/pytorch/module/linear.py:431-440`

backward 里恢复出来后变量名叫 `weight_fp8`。这里不是重新生成了另一种概念上的 Tensor，而是 forward 保存的 quantized weight workspace 在 backward 里的名字：

```python
inputmat, weight_fp8, weight, bias = restore_from_saved(
    ctx.tensor_objects, saved_tensors
)
```

证据：`../TransformerEngine/transformer_engine/pytorch/module/linear.py:504-508`

dgrad 前，TE 要求这个 `weight_fp8` 具有 columnwise：

```python
weight_fp8.update_usage(columnwise_usage=True)
```

证据：`../TransformerEngine/transformer_engine/pytorch/module/linear.py:690-697`

关键差异在这里：input 在 forward 后有明确的 `rowwise_usage=False`；weight 没有对应的裁剪逻辑。源码里没有对 `weightmat` 做类似下面这样的释放：

```python
weightmat.update_usage(rowwise_usage=False, columnwise_usage=True)
```

实际看到的是 forward 侧确保 rowwise 可用：

```python
weightmat.update_usage(rowwise_usage=True)
```

证据：`../TransformerEngine/transformer_engine/pytorch/module/linear.py:278`

这和 weight workspace 的缓存复用语义一致。`get_weight_workspace()` 取缓存时会检查缓存里是否存在当前 quantizer 要求的 usage；如果缺少 rowwise 或 columnwise，就 reset cache：

```python
elif isinstance(out, MXFP8TensorStorage):
    if quantizer.rowwise_usage and out._rowwise_data is None:
        reset_cache = True
    elif quantizer.columnwise_usage and out._columnwise_data is None:
        reset_cache = True
```

证据：`../TransformerEngine/transformer_engine/pytorch/module/base.py:1377-1391`

所以 weight 的生命周期可以概括为：

```text
forward quantize:
  weight rowwise
  如果需要 dgrad，同时准备 weight columnwise

forward GEMM:
  消费 weight rowwise

save-for-backward:
  保存 weightmat 这个 quantized weight workspace

backward dgrad:
  weightmat 恢复为 weight_fp8
  消费 weight columnwise

后续 forward / microbatch:
  weight workspace 可复用
  源码没有像 input 一样主动丢弃 rowwise
```

结论：训练且需要 dgrad 时，weight workspace 通常会同时缓存 rowwise + columnwise。forward 后反向确实只需要 weight columnwise 做 dgrad，但 TE 当前实现没有在 save-for-backward 前丢掉 weight rowwise；它把 `weightmat` 作为可复用 workspace 保存下来。

## 6. 显存结论

MXFP8 的 rowwise / columnwise 是两套 `data + scale_inv`。只要一个 tensor 同时持有两种 usage，就会带来额外显存。

但 input 和 weight 要分开看：

```text
input:
  forward 需要 rowwise
  backward wgrad 需要 columnwise
  TE 在 forward 后会主动丢弃 rowwise
  因此 input 的双份缓存更偏短生命周期，源码有裁剪逻辑

weight:
  forward 需要 rowwise
  backward dgrad 需要 columnwise
  weight workspace 需要跨 forward/backward，并可能跨 microbatch/iteration 复用
  TE 当前实现通常保留 rowwise + columnwise
  因此 weight 的额外缓存显存更明显
```

所以更准确的表述是：

```text
MXFP8 不是简单“显存一定下降”。
它降低的是 GEMM 输入侧数据位宽和关键路径带宽压力；
但为了满足不同 GEMM layout，TE 需要 rowwise / columnwise 两套方向相关的量化表示。
对 input，TE 会尽量在 forward 后裁剪不再需要的方向；
对 weight，TE 为复用 workspace 通常保留两套表示。
```

这也是为什么讨论 MXFP8 显存时要区分：

```text
参数本体 / master weight
weight 量化 workspace
activation/input 保存给 backward 的缓存
grad_output 缓存
scale_inv 元数据
```

单看 weight MXFP8 workspace，rowwise + columnwise 的确会增加显存；端到端是否省显存则取决于这些缓存、原始参数、优化器状态、activation 保存策略和通信路径的综合结果。
