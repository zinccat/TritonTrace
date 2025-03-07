# From: 100_ConvTranspose3d_Clamp_Min_Divide

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_convolution_backward_2(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 16
    rnumel = 21
    RBLOCK: tl.constexpr = 32

    # Calculate the starting index for the input data
    input_offset = tl.program_id(0) * XBLOCK
    input_indices = input_offset + tl.arange(0, XBLOCK)[:, None]

    # Create a mask to ensure indices are within bounds
    input_mask = input_indices < xnumel
    reduction_indices = tl.arange(0, RBLOCK)[None, :]
    reduction_mask = reduction_indices < rnumel

    # Calculate the indices for loading data
    reduction_index = reduction_indices
    input_index = input_indices

    # Load data with masking and broadcast
    loaded_data = tl.load(in_ptr0 + (reduction_index + 21 * input_index), reduction_mask & input_mask, other=0.0)
    broadcasted_data = tl.broadcast_to(loaded_data, [XBLOCK, RBLOCK])

    # Apply mask and sum along the reduction dimension
    masked_data = tl.where(reduction_mask & input_mask, broadcasted_data, 0)
    summed_data = tl.sum(masked_data, 1)[:, None]

    # Store the result back to the output pointer
    tl.store(out_ptr0 + (input_index), summed_data, input_mask)