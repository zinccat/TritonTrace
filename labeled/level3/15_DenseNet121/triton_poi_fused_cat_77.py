# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_77poi_fused_cat_77(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 564480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel

    # Calculate indices for accessing input pointers
    channel_index = (xindex // 196) % 288
    row_index = xindex % 196
    batch_index = xindex // 56448
    flat_index = xindex

    # Temporary variables for conditions
    max_channel_index = 256
    tmp_channel_index = channel_index
    is_within_first_input = tmp_channel_index < max_channel_index

    # Load from the first input pointer
    load_mask_first_input = is_within_first_input & xmask
    value_from_first_input = tl.load(in_ptr0 + (row_index + 196 * channel_index + 50176 * batch_index), load_mask_first_input, other=0.0)

    # Load from the second input pointer
    is_within_second_input = tmp_channel_index >= max_channel_index
    load_mask_second_input = is_within_second_input & xmask
    value_from_second_input = tl.load(in_ptr1 + (row_index + 196 * ((-256) + channel_index) + 6272 * batch_index), load_mask_second_input, other=0.0)

    # Select value based on channel index
    selected_value = tl.where(is_within_first_input, value_from_first_input, value_from_second_input)

    # Store the result in the output pointer
    tl.store(out_ptr0 + (flat_index), selected_value, xmask)