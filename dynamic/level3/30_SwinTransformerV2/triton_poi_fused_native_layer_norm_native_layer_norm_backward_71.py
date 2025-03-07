# From: 30_SwinTransformerV2

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32', 10: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_native_layer_norm_backward_71', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 384) % 196
    x2 = (xindex // 75264)
    x3 = xindex
    x4 = (xindex // 384)
    x0 = xindex % 384
    tmp0 = tl.load(in_ptr0 + (x1 + (98*ks0*x2)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp6 = tl.load(in_ptr2 + (x4), xmask, eviction_policy='evict_last')
    tl.device_assert(((0 <= tl.where((((x1 % (7*ks0)) + (triton_helpers.remainder_integer((-3) + (7*ks0), 7*ks0))) % (7*ks0)) < 0, 14 + (((x1 % (7*ks0)) + (triton_helpers.remainder_integer((-3) + (7*ks0), 7*ks0))) % (7*ks0)), ((x1 % (7*ks0)) + (triton_helpers.remainder_integer((-3) + (7*ks0), 7*ks0))) % (7*ks0))) & (tl.where((((x1 % (7*ks0)) + (triton_helpers.remainder_integer((-3) + (7*ks0), 7*ks0))) % (7*ks0)) < 0, 14 + (((x1 % (7*ks0)) + (triton_helpers.remainder_integer((-3) + (7*ks0), 7*ks0))) % (7*ks0)), ((x1 % (7*ks0)) + (triton_helpers.remainder_integer((-3) + (7*ks0), 7*ks0))) % (7*ks0)) < 14)) | ~(xmask), "index out of bounds: 0 <= tl.where((((x1 % (7*ks0)) + (triton_helpers.remainder_integer((-3) + (7*ks0), 7*ks0))) % (7*ks0)) < 0, 14 + (((x1 % (7*ks0)) + (triton_helpers.remainder_integer((-3) + (7*ks0), 7*ks0))) % (7*ks0)), ((x1 % (7*ks0)) + (triton_helpers.remainder_integer((-3) + (7*ks0), 7*ks0))) % (7*ks0)) < 14")
    tmp9 = tl.load(in_ptr3 + ((((x3 // 384) % 196) // (7*ks0)) % 14), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1 + (98*ks0*x2)), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x4), xmask, eviction_policy='evict_last')
    tmp1 = 0.0026041666666666665
    tmp2 = tmp0 * tmp1
    tmp4 = 384.0
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 - tmp6
    tmp10 = tl.full([XBLOCK], 14, tl.int32)
    tmp11 = tmp9 + tmp10
    tmp12 = tmp9 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp9)
    tl.device_assert(((0 <= tmp13) & (tmp13 < 14)) | ~(xmask), "index out of bounds: 0 <= tmp13 < 14")
    tmp15 = tl.load(in_ptr4 + (x0 + (384*((tl.where((((x1 % (7*ks0)) + (triton_helpers.remainder_integer((-3) + (7*ks0), 7*ks0))) % (7*ks0)) < 0, 14 + (((x1 % (7*ks0)) + (triton_helpers.remainder_integer((-3) + (7*ks0), 7*ks0))) % (7*ks0)), ((x1 % (7*ks0)) + (triton_helpers.remainder_integer((-3) + (7*ks0), 7*ks0))) % (7*ks0))) % 7)) + (2688*(tmp13 % 7)) + (18816*(((tl.where((((x1 % (7*ks0)) + (triton_helpers.remainder_integer((-3) + (7*ks0), 7*ks0))) % (7*ks0)) < 0, 14 + (((x1 % (7*ks0)) + (triton_helpers.remainder_integer((-3) + (7*ks0), 7*ks0))) % (7*ks0)), ((x1 % (7*ks0)) + (triton_helpers.remainder_integer((-3) + (7*ks0), 7*ks0))) % (7*ks0))) // 7) % ks0)) + (18816*ks0*((tmp13 // 7) % 2)) + (18816*x2*(triton_helpers.div_floor_integer(ks1,  libdevice.trunc((ks1.to(tl.float64)) / 4.00000000000000).to(tl.int32))))), xmask)
    tmp17 = tmp15 - tmp16
    tmp18 = tmp17 * tmp0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp7 - tmp20
    tmp22 = tmp2 * tmp21
    tl.store(out_ptr0 + (x3), tmp22, xmask)
