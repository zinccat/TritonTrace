# From: 29_SwinMLP

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_slice_backward_43', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 5376) % 28
    x1 = (xindex // 192) % 28
    x0 = xindex % 192
    x3 = (xindex // 150528)
    x6 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = ((-1) + x2) % 2
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 == tmp4
    tmp6 = tmp2 & tmp5
    tmp7 = x1
    tmp8 = tmp7 >= tmp1
    tmp9 = ((-1) + x1) % 2
    tmp10 = tmp9 == tmp4
    tmp11 = tmp8 & tmp10
    tmp12 = tmp11 & tmp6
    tmp13 = tl.load(in_ptr0 + (576 + x0 + (768*(triton_helpers.div_floor_integer((-1) + x1,  2))) + (10752*(triton_helpers.div_floor_integer((-1) + x2,  2))) + (150528*x3)), tmp12 & xmask, other=0.0)
    tmp14 = 0.0
    tmp15 = tl.where(tmp11, tmp13, tmp14)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp6, tmp15, tmp16)
    tmp18 = tl.where(tmp6, tmp17, tmp14)
    tmp19 = ((x6 // 5376) % 28) % 2
    tmp20 = tmp19 == tmp4
    tmp21 = tmp11 & tmp20
    tmp22 = tl.load(in_ptr0 + (384 + x0 + (768*(triton_helpers.div_floor_integer((-1) + x1,  2))) + (10752*(x2 // 2)) + (150528*x3)), tmp21 & xmask, other=0.0)
    tmp23 = tl.where(tmp11, tmp22, tmp14)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp20, tmp23, tmp24)
    tmp26 = tl.where(tmp20, tmp25, tmp14)
    tmp27 = tmp18 + tmp26
    tmp28 = ((x6 // 192) % 28) % 2
    tmp29 = tmp28 == tmp4
    tmp30 = tmp29 & tmp6
    tmp31 = tl.load(in_ptr0 + (192 + x0 + (768*(x1 // 2)) + (10752*(triton_helpers.div_floor_integer((-1) + x2,  2))) + (150528*x3)), tmp30 & xmask, other=0.0)
    tmp32 = tl.where(tmp29, tmp31, tmp14)
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp6, tmp32, tmp33)
    tmp35 = tl.where(tmp6, tmp34, tmp14)
    tmp36 = tmp27 + tmp35
    tmp37 = tmp29 & tmp20
    tmp38 = tl.load(in_ptr0 + (x0 + (768*(x1 // 2)) + (10752*(x2 // 2)) + (150528*x3)), tmp37 & xmask, other=0.0)
    tmp39 = tl.where(tmp29, tmp38, tmp14)
    tmp40 = tl.full(tmp39.shape, 0.0, tmp39.dtype)
    tmp41 = tl.where(tmp20, tmp39, tmp40)
    tmp42 = tl.where(tmp20, tmp41, tmp14)
    tmp43 = tmp36 + tmp42
    tl.store(in_out_ptr0 + (x6), tmp43, xmask)
