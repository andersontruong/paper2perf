import math

import torch

aten = torch.ops.aten

requirement_dict = {
    'exp': 0,
    'alu': 0,
    'mac': 0
}
def req_to_vec(req_dict) -> list:
    merged = {**requirement_dict, **req_dict}
    req = [merged[k] for k in sorted(merged.keys())]
    req = [r._sympy_() if isinstance(r, torch.SymInt) else r for r in req]
    return req

def softmax_req(a_shape, dim: int, half_to_float: bool, **kwargs) -> list:
    n_elems = math.prod(a_shape)
    req_dict = requirement_dict.copy()

    # Online softmax algorithm
    req_dict['exp'] = 3*n_elems
    req_dict['alu'] = 7*n_elems
    req_vec = req_to_vec(req_dict)

    return req_vec

def gelu_req(a_shape, approximate: str = "none", **kwargs) -> list:
    """Hardware requirement vector for GELU activation.

    GELU(x) = x * Φ(x) where Φ(x) is the CDF of standard normal distribution.
    Approximate form uses tanh approximation requiring 10 ALU + 1 EXP per element.
    """
    n_elems = math.prod(a_shape)
    req_dict = requirement_dict.copy()
    req_dict['exp'] = 1 * n_elems
    req_dict['alu'] = 10 * n_elems
    return req_to_vec(req_dict)

def native_layer_norm_req(input_shape, normalized_shape, weight_shape=None, bias_shape=None, eps=None, out_shape=None, **kwargs) -> list:
    """Hardware requirement vector for native_layer_norm.

    Layer norm requires computing mean, variance, normalization, and affine transform:
    - Mean: sum and divide
    - Variance: sum of squared diffs, divide
    - Normalize: subtract mean, divide by std
    - Affine: multiply weight, add bias
    Total: 7 ALU ops per element.
    """
    n_elems = math.prod(input_shape)
    req_dict = requirement_dict.copy()
    req_dict['alu'] = 7 * n_elems
    return req_to_vec(req_dict)

def mm_req(a_shape, b_shape, *args, out_shape=None, **kwargs) -> list:
    """Hardware requirement vector for mm (matrix multiply)."""
    m, k = a_shape
    _k2, n = b_shape
    req_dict = requirement_dict.copy()
    # PyTorch uses 2*m*n*k FLOPs, mac = FLOPs/2
    req_dict['mac'] = m * n * k
    return req_to_vec(req_dict)

def addmm_req(self_shape, a_shape, b_shape, out_shape=None, **kwargs) -> list:
    """Hardware requirement vector for addmm (add matrix multiply)."""
    return mm_req(a_shape, b_shape)

def bmm_req(a_shape, b_shape, out_shape=None, **kwargs) -> list:
    """Hardware requirement vector for bmm (batch matrix multiply)."""
    b, m, k = a_shape
    _b2, _k2, n = b_shape
    req_dict = requirement_dict.copy()
    req_dict['mac'] = b * m * n * k
    return req_to_vec(req_dict)

def baddbmm_req(self_shape, a_shape, b_shape, out_shape=None, **kwargs) -> list:
    """Hardware requirement vector for baddbmm (batch add mm)."""
    return bmm_req(a_shape, b_shape)

def _scaled_mm_req(
    a_shape,
    b_shape,
    scale_a_shape,
    scale_b_shape,
    bias_shape=None,
    scale_result_shape=None,
    out_dtype=None,
    use_fast_accum=False,
    out_shape=None,
    **kwargs,
) -> list:
    """Hardware requirement vector for _scaled_mm."""
    return mm_req(a_shape, b_shape)

def conv_req(x_shape, w_shape, _bias, _stride, _padding, _dilation, transposed,
             *args, out_shape=None, **kwargs) -> list:
    """Hardware requirement vector for convolution."""
    batch_size = x_shape[0]
    conv_shape = (x_shape if transposed else out_shape)[2:]
    c_out, c_in, *filter_size = w_shape
    mac_count = math.prod(conv_shape) * math.prod(filter_size) * batch_size * c_out * c_in
    req_dict = requirement_dict.copy()
    req_dict['mac'] = mac_count
    return req_to_vec(req_dict)

def conv_backward_req(
    grad_out_shape,
    x_shape,
    w_shape,
    _bias,
    _stride,
    _padding,
    _dilation,
    transposed,
    _output_padding,
    _groups,
    output_mask,
    out_shape
) -> list:
    """Hardware requirement vector for convolution backward.

    PyTorch counts: grad_input + grad_weight (+ grad_bias - mac=0)
    """
    def t(shape): return [shape[1], shape[0]] + list(shape[2:])
    flop_count = 0

    # grad_inp as conv_transpose(grad_out, weight)
    if output_mask[0]:
        grad_input_shape = out_shape[0] if isinstance(out_shape, (list, tuple)) else out_shape
        # conv_flop_count with transposed=not transposed
        batch_size = grad_out_shape[0]
        conv_shape = (grad_input_shape if transposed else grad_out_shape)[2:]
        c_out, c_in, *filter_size = w_shape
        flop_count += math.prod(conv_shape) * math.prod(filter_size) * batch_size * c_out * c_in * 2

    # grad_weight
    if output_mask[1]:
        if transposed:
            # grad_weight of transposed conv as conv(grad_out, inp)
            batch_size = grad_out_shape[0]
            c_out = grad_out_shape[1]
            c_in = x_shape[1]
            conv_shape = x_shape[2:]
            filter_size = grad_out_shape[2:]
            flop_count += math.prod(conv_shape) * math.prod(filter_size) * batch_size * c_out * c_in * 2
        else:
            # grad_weight as conv(inp, grad_out)
            batch_size = x_shape[0]
            c_out = w_shape[0]
            c_in = x_shape[1]
            conv_shape = out_shape[1][2:] if isinstance(out_shape, (list, tuple)) else grad_out_shape[2:]
            filter_size = w_shape[2:]
            flop_count += math.prod(conv_shape) * math.prod(filter_size) * batch_size * c_out * c_in * 2

    # PyTorch flop_count / 2 = mac count
    req_dict = requirement_dict.copy()
    req_dict['mac'] = flop_count // 2
    return req_to_vec(req_dict)

def sdpa_forward_req(query_shape, key_shape, value_shape, *args, out_shape=None, **kwargs) -> list:
    """Hardware requirement vector for scaled dot-product attention forward.

    Two bmm ops:
    1. Q @ K: (b*h_q, s_q, d_q) @ (b*h_q, d_q, s_k) -> scores
    2. scores @ V: (b*h_q, s_q, s_k) @ (b*h_q, s_k, d_v) -> out
    """
    b, h_q, s_q, d_q = query_shape
    _b2, _h_kv, s_k, _d2 = key_shape
    _b3, _h3, _s3, d_v = value_shape

    req_dict = requirement_dict.copy()
    mac_count = 0
    # Q @ K.T: b*h_q*s_q*d_q*s_k / 2 (flops/2 = mac)
    mac_count += b * h_q * s_q * d_q * s_k
    # scores @ V: b*h_q*s_q*s_k*d_v / 2
    mac_count += b * h_q * s_q * s_k * d_v
    req_dict['mac'] = mac_count
    return req_to_vec(req_dict)

def sdpa_backward_req(grad_out_shape, query_shape, key_shape, value_shape, *args, out_shape=None, **kwargs) -> list:
    """Hardware requirement vector for SDPA backward (includes recompute).

    Step 1: Recompute scores matrix
    Step 2-6: 5 bmm ops for gradient propagation
    """
    b, h_q, s_q, d_q = query_shape
    _b2, _h_kv, s_k, _d2 = key_shape
    _b3, _h3, _s3, d_v = value_shape
    _b4, _h4, _s4, _d4 = grad_out_shape

    req_dict = requirement_dict.copy()
    mac_count = 0
    b_h = b * h_q

    # Step 1: Recompute Q @ K (s_q, d_q) @ (d_q, s_k) -> scores
    mac_count += b_h * s_q * d_q * s_k

    # Step 2: gradOut @ V.T: (s_q, d_v) @ (d_v, s_k) -> gradScores
    mac_count += b_h * s_q * d_v * s_k

    # Step 3: scores.T @ gradOut: (s_k, s_q) @ (s_q, d_v) -> gradV
    mac_count += b_h * s_k * s_q * d_v

    # Step 4: gradScores @ K: (s_q, s_k) @ (s_k, d_q) -> gradQ
    mac_count += b_h * s_q * s_k * d_q

    # Step 5: Q.T @ gradScores: (d_q, s_q) @ (s_q, s_k) -> gradK
    mac_count += b_h * d_q * s_q * s_k

    req_dict['mac'] = mac_count
    return req_to_vec(req_dict)

def sum_req(input_shape, dim, *args, out_shape=None, **kwargs) -> list:
    """Hardware requirement vector for a sum reduction.

    Summing a reduced extent of ``r`` values down to one costs ``r - 1`` adds
    per output element, i.e. ``(r - 1) * output_elements`` adds in total. This
    also covers the tile-join reduction a contracting axis lowers to.
    """
    ndim = len(input_shape)
    dims = dim if isinstance(dim, (list, tuple)) else [dim]
    reduced_dims = {axis % ndim for axis in dims}
    reduced = 1
    output_elements = 1
    for axis, extent in enumerate(input_shape):
        if axis in reduced_dims:
            reduced = reduced * extent
        else:
            output_elements = output_elements * extent
    req_dict = requirement_dict.copy()
    req_dict['alu'] = (reduced - 1) * output_elements
    return req_to_vec(req_dict)

flop_requirement_vectors = {
    aten._softmax: softmax_req,
    aten.gelu: gelu_req,
    aten.native_layer_norm: native_layer_norm_req,
    aten.mm: mm_req,
    aten.addmm: addmm_req,
    aten.bmm: bmm_req,
    aten.baddbmm: baddbmm_req,
    aten._scaled_mm: _scaled_mm_req,
    aten.sum: sum_req,
    aten.convolution: conv_req,
    aten._convolution: conv_req,
    aten.cudnn_convolution: conv_req,
    aten._slow_conv2d_forward: conv_req,
    aten.convolution_overrideable: conv_req,
    aten.convolution_backward: conv_backward_req,
    aten._scaled_dot_product_efficient_attention: sdpa_forward_req,
    aten._scaled_dot_product_flash_attention: sdpa_forward_req,
    aten._scaled_dot_product_cudnn_attention: sdpa_forward_req,
    aten._scaled_dot_product_efficient_attention_backward: sdpa_backward_req,
    aten._scaled_dot_product_flash_attention_backward: sdpa_backward_req,
    aten._scaled_dot_product_cudnn_attention_backward: sdpa_backward_req,
}
