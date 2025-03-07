# From: 19_MobileNetV1

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32', 9: 'i32', 10: 'i32', 11: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_backward_native_batch_norm_backward_threshold_backward_5', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x0 = xindex % ks0
    x1 = (xindex // ks0) % ks0
    x5 = (xindex // ks1)
    x2 = (xindex // ks1) % 1024
    tmp0 = tl.load(in_ptr0 + (x4), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x5 + (x5*((triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks2,  32)),  7))*(triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks2,  32)),  7)))) + ((triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks2,  32)),  7))*((((0) * ((0) >= ((x1 // 7))) + ((x1 // 7)) * (((x1 // 7)) > (0)))) * ((((0) * ((0) >= ((x1 // 7))) + ((x1 // 7)) * (((x1 // 7)) > (0)))) <= ((-1) + ((1 + (x1 // 7)) * ((1 + (x1 // 7)) <= (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks2,  32)),  7)))) + (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks2,  32)),  7))) * ((1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks2,  32)),  7))) < (1 + (x1 // 7)))))) + ((-1) + ((1 + (x1 // 7)) * ((1 + (x1 // 7)) <= (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks2,  32)),  7)))) + (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks2,  32)),  7))) * ((1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks2,  32)),  7))) < (1 + (x1 // 7))))) * (((-1) + ((1 + (x1 // 7)) * ((1 + (x1 // 7)) <= (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks2,  32)),  7)))) + (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks2,  32)),  7))) * ((1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks2,  32)),  7))) < (1 + (x1 // 7))))) < (((0) * ((0) >= ((x1 // 7))) + ((x1 // 7)) * (((x1 // 7)) > (0))))))) + (2*x5*(triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks2,  32)),  7))) + ((((0) * ((0) >= ((x0 // 7))) + ((x0 // 7)) * (((x0 // 7)) > (0)))) * ((((0) * ((0) >= ((x0 // 7))) + ((x0 // 7)) * (((x0 // 7)) > (0)))) <= ((-1) + ((1 + (x0 // 7)) * ((1 + (x0 // 7)) <= (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks2,  32)),  7)))) + (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks2,  32)),  7))) * ((1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks2,  32)),  7))) < (1 + (x0 // 7)))))) + ((-1) + ((1 + (x0 // 7)) * ((1 + (x0 // 7)) <= (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks2,  32)),  7)))) + (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks2,  32)),  7))) * ((1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks2,  32)),  7))) < (1 + (x0 // 7))))) * (((-1) + ((1 + (x0 // 7)) * ((1 + (x0 // 7)) <= (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks2,  32)),  7)))) + (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks2,  32)),  7))) * ((1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks2,  32)),  7))) < (1 + (x0 // 7))))) < (((0) * ((0) >= ((x0 // 7))) + ((x0 // 7)) * (((x0 // 7)) > (0)))))) + ((((0) * ((0) >= ((x1 // 7))) + ((x1 // 7)) * (((x1 // 7)) > (0)))) * ((((0) * ((0) >= ((x1 // 7))) + ((x1 // 7)) * (((x1 // 7)) > (0)))) <= ((-1) + ((1 + (x1 // 7)) * ((1 + (x1 // 7)) <= (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks2,  32)),  7)))) + (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks2,  32)),  7))) * ((1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks2,  32)),  7))) < (1 + (x1 // 7)))))) + ((-1) + ((1 + (x1 // 7)) * ((1 + (x1 // 7)) <= (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks2,  32)),  7)))) + (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks2,  32)),  7))) * ((1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks2,  32)),  7))) < (1 + (x1 // 7))))) * (((-1) + ((1 + (x1 // 7)) * ((1 + (x1 // 7)) <= (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks2,  32)),  7)))) + (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks2,  32)),  7))) * ((1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks2,  32)),  7))) < (1 + (x1 // 7))))) < (((0) * ((0) >= ((x1 // 7))) + ((x1 // 7)) * (((x1 // 7)) > (0))))))), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr2 + (x4), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 / 49
    tmp5 = ((0) * ((0) >= ((x1 // 7))) + ((x1 // 7)) * (((x1 // 7)) > (0)))
    tmp6 = ((1 + (x1 // 7)) * ((1 + (x1 // 7)) <= (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks2,  32)),  7)))) + (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks2,  32)),  7))) * ((1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks2,  32)),  7))) < (1 + (x1 // 7))))
    tmp7 = tmp5 < tmp6
    tmp8 = ((0) * ((0) >= ((x0 // 7))) + ((x0 // 7)) * (((x0 // 7)) > (0)))
    tmp9 = ((1 + (x0 // 7)) * ((1 + (x0 // 7)) <= (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks2,  32)),  7)))) + (1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks2,  32)),  7))) * ((1 + (triton_helpers.div_floor_integer((-6) + (triton_helpers.div_floor_integer((-1) + ks2,  32)),  7))) < (1 + (x0 // 7))))
    tmp10 = tmp8 < tmp9
    tmp11 = tmp7 & tmp10
    tmp12 = tl.where(tmp11, tmp4, tmp1)
    tmp13 = tl.where(tmp2, tmp1, tmp12)
    tmp16 = tmp14 - tmp15
    tmp18 = 1.00000000000000 / (((1024*ks3) + (1024*ks3*((triton_helpers.div_floor_integer((-1) + ks2,  32))*(triton_helpers.div_floor_integer((-1) + ks2,  32)))) + (2048*ks3*(triton_helpers.div_floor_integer((-1) + ks2,  32)))) / 1024)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 * tmp19
    tmp22 = tmp21 * tmp21
    tmp23 = tmp20 * tmp22
    tmp24 = tmp16 * tmp23
    tmp25 = tmp13 - tmp24
    tl.store(out_ptr0 + (x4), tmp25, xmask)
