# From: 15_DenseNet121

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_133', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex % ks0
    x3 = (xindex // ks0)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (416 + x2 + (416*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)))) + (512*x3) + (832*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2))) + (512*x3*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)))) + (1024*x3*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)))), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (416 + x2 + (416*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)))) + (480*x3) + (832*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2))) + (480*x3*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)))) + (960*x3*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)))), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (416 + x2 + (416*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)))) + (448*x3) + (832*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2))) + (448*x3*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)))) + (896*x3*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)))), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x4), tmp4, xmask)
