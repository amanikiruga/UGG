"""Small GPU checks for UGG's FlashAttention-2 forward and backward paths.

    CUDA_VISIBLE_DEVICES=0 python tests/test_flash_attention.py

Requires FlashAttention-2; fails instead of using the native attention fallback.
Compares both supported dtypes against PyTorch's float32 math attention.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from algorithms.wan.modules import attention  # noqa: E402


def reference(q, k, v, lengths, causal, window):
    outputs = []
    for i, nk in enumerate(lengths):
        nq = q.shape[1]
        qi, ki, vi = [x[i, :n].transpose(0, 1)[None]
                      for x, n in ((q, nq), (k, nk), (v, nk))]
        mask = None
        if causal or window != (-1, -1):
            center = torch.arange(nq, device=q.device)[:, None] + nk - nq
            keys = torch.arange(nk, device=q.device)[None, :]
            mask = torch.ones((nq, nk), device=q.device, dtype=torch.bool)
            if causal:
                mask &= keys <= center
            if window[0] >= 0:
                mask &= keys >= center - window[0]
            if window[1] >= 0:
                mask &= keys <= center + window[1]
        with sdpa_kernel(SDPBackend.MATH):
            out = F.scaled_dot_product_attention(qi, ki, vi, attn_mask=mask)
        outputs.append(out[0].transpose(0, 1))
    return torch.stack(outputs)


def compare(actual, expected, dtype):
    actual = actual.float()
    assert torch.isfinite(actual).all()
    relative = ((actual - expected).norm() / expected.norm().clamp_min(1e-8)).item()
    limit = 0.02 if dtype == torch.bfloat16 else 0.003
    assert relative < limit, (relative, limit)
    torch.testing.assert_close(actual, expected, rtol=limit,
                               atol=0.03 if dtype == torch.bfloat16 else 0.005)
    return relative


def main():
    assert torch.cuda.is_available(), 'CUDA is required'
    assert attention.FLASH_ATTN_2_AVAILABLE, 'FlashAttention-2 must be installed'
    print('GPU:', torch.cuda.get_device_name(0))
    print('PyTorch:', torch.__version__, 'FlashAttention:', attention.flash_attn.__version__)
    cases = [('self', 32, [32, 32], False, (-1, -1)),
             ('cross_padding', 48, [48, 37], False, (-1, -1)),
             ('causal', 48, [48, 37], True, (-1, -1)),
             ('window', 48, [48, 37], False, (3, 2))]
    for dtype in (torch.float16, torch.bfloat16):
        for name, nk, lengths, causal, window in cases:
            torch.manual_seed(0)
            inputs = [torch.randn(2, n, 4, 128, device='cuda', dtype=dtype, requires_grad=True)
                      for n in (32, nk, nk)]
            refs = [x.detach().float().requires_grad_(True) for x in inputs]
            out = attention.flash_attention(*inputs, k_lens=torch.tensor(lengths),
                                             causal=causal, window_size=window,
                                             deterministic=True, dtype=dtype, version=2)
            expected = reference(*refs, lengths, causal, window)
            forward_error = compare(out, expected, dtype)
            grad = torch.randn_like(out).float()
            (out.float() * grad).sum().backward()
            (expected * grad).sum().backward()
            gradient_error = max(compare(x.grad, y.grad, dtype) for x, y in zip(inputs, refs))
            print(f'{dtype} {name}: forward relative error={forward_error:.5f}, '
                  f'gradient relative error={gradient_error:.5f}')
    print('FLASHATTENTION-2 KERNEL CHECKS PASS (8 forward/backward cases)')


if __name__ == '__main__':
    main()
