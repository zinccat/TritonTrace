# From: 47_Conv3d_Mish_Tanh

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_convolution_backward_2(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 16
    rnumel = 25
    RBLOCK: tl.constexpr = 32

    # Calculate the starting index for the current program
    x_start_index = tl.program_id(0) * XBLOCK
    x_indices = x_start_index + tl.arange(0, XBLOCK)[:, None]

    # Create masks for valid indices
    x_valid_mask = x_indices < xnumel
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_valid_mask = r_indices < rnumel

    # Load data with masking
    r_indices_adjusted = r_indices
    x_indices_adjusted = x_indices
    loaded_data = tl.load(in_ptr0 + (r_indices_adjusted + 25 * x_indices_adjusted), r_valid_mask & x_valid_mask, other=0.0)

    # Broadcast loaded data to the required shape
    broadcasted_data = tl.broadcast_to(loaded_data, [XBLOCK, RBLOCK])

    # Apply mask and zero out invalid entries
    masked_data = tl.where(r_valid_mask & x_valid_mask, broadcasted_data, 0)

    # Sum along the first dimension and reshape
    summed_data = tl.sum(masked_data, 1)[:, None]

    # Store the result back to the output pointer
    tl.store(out_ptr0 + (x_indices_adjusted), summed_data, x_valid_mask)