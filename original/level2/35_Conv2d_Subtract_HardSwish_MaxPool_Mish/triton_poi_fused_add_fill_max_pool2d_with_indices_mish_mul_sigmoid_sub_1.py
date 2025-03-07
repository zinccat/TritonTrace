# From: 35_Conv2d_Subtract_HardSwish_MaxPool_Mish

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i8', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_max_pool2d_with_indices_mish_mul_sigmoid_sub_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 460800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 15
    x3 = (xindex // 15)
    x2 = (xindex // 3600)
    x4 = xindex % 3600
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (60*x3)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (60*x3)), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (30 + (2*x0) + (60*x3)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (31 + (2*x0) + (60*x3)), None, eviction_policy='evict_last')
    tmp2 = tmp1 > tmp0
    tmp3 = tl.full([1], 1, tl.int8)
    tmp4 = tl.full([1], 0, tl.int8)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp8 = tmp7 > tmp6
    tmp9 = tl.full([1], 2, tl.int8)
    tmp10 = tl.where(tmp8, tmp9, tmp5)
    tmp11 = triton_helpers.maximum(tmp7, tmp6)
    tmp13 = tmp12 > tmp11
    tmp14 = tl.full([1], 3, tl.int8)
    tmp15 = tl.where(tmp13, tmp14, tmp10)
    tmp16 = triton_helpers.maximum(tmp12, tmp11)
    tmp17 = 20.0
    tmp18 = tmp16 > tmp17
    tmp19 = tl_math.exp(tmp16)
    tmp20 = libdevice.log1p(tmp19)
    tmp21 = tl.where(tmp18, tmp16, tmp20)
    tmp22 = libdevice.tanh(tmp21)
    tmp23 = tmp16 * tmp22
    tmp24 = tl.sigmoid(tmp16)
    tmp25 = tmp16 * tmp24
    tmp26 = tmp22 * tmp22
    tmp27 = 1.0
    tmp28 = tmp27 - tmp26
    tmp29 = tmp25 * tmp28
    tmp30 = tmp22 + tmp29
    tl.store(out_ptr0 + (x4 + (3712*x2)), tmp15, None)
    tl.store(out_ptr1 + (x5), tmp23, None)
    tl.store(out_ptr2 + (x4 + (3616*x2)), tmp30, None)
