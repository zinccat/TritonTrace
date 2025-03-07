# From: 51_Gemm_Subtract_GlobalAvgPool_LogSumExp_GELU_ResidualAdd

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_addmm_mean_sub_0(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel
):
    XBLOCK: tl.constexpr = 1
    RBLOCK: tl.constexpr = 512

    # Calculate the offset and index for the X dimension
    x_offset = tl.program_id(0) * XBLOCK
    x_index = tl.full([1], x_offset, tl.int32)

    # Initialize rindex for the R dimension
    r_index = tl.arange(0, RBLOCK)[:]

    # Load data from input pointers
    input_data0 = tl.load(in_ptr0 + (r_index + 512 * x_index), None)
    input_data1 = tl.load(in_ptr1 + (r_index), None, eviction_policy='evict_last')
    input_data2 = tl.load(in_ptr2 + (r_index), None, eviction_policy='evict_last')

    # Perform element-wise operations
    sum_data = input_data0 + input_data1
    subtracted_data = sum_data - input_data2

    # Broadcast and sum the results
    broadcasted_data = tl.broadcast_to(subtracted_data, [RBLOCK])
    summed_result = triton_helpers.promote_to_tensor(tl.sum(broadcasted_data, 0))

    # Calculate the mean
    num_elements = 512.0
    mean_result = summed_result / num_elements

    # Store the result
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x_index), mean_result, None)