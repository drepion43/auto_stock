"""Mamba(SSM)의 선택적 스캔(selective scan) 병렬 구현(Blelloch 방식) — PyTorch autograd Function.

`MambaStock-main/pscan.py`(원본, https://github.com/alxndrTL/mamba.py 기반)에서 그대로
가져왔다. 로직은 수정하지 않았다 — mamba_block.py가 `config.pscan=True`일 때 이 함수를
호출해 순차 스캔(O(L))보다 빠른 O(log L) 병렬 스캔을 쓴다."""

import math

import torch
import torch.nn.functional as F


def npo2(length: int) -> int:
    """다음 2의 거듭제곱."""
    return 2 ** math.ceil(math.log2(length))


def pad_npo2(X: torch.Tensor) -> torch.Tensor:
    """길이(dim=1)를 다음 2의 거듭제곱으로 패딩한다.

    Args:
        X : (B, L, D, N)

    Returns:
        Y : (B, npo2(L), D, N)
    """
    len_npo2 = npo2(X.size(1))
    pad_tuple = (0, 0, 0, 0, 0, len_npo2 - X.size(1))
    return F.pad(X, pad_tuple, "constant", 0)


class PScan(torch.autograd.Function):
    @staticmethod
    def pscan(A, X):
        # A : (B, D, L, N)
        # X : (B, D, L, N)

        # X를 제자리에서(in-place) 병렬 스캔으로 채운다.
        # H[t] = A[t] * H[t-1] + X[t], H[0] = 0을 병렬로 계산한다(순차 T단계 대신 약 2*log2(T)단계).
        # L이 2의 거듭제곱일 때만 지원한다(코드를 더 명확하게 하기 위함).

        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        # up sweep (마지막 2단계는 풀어서 처리)
        Aa = A
        Xa = X
        for _ in range(num_steps - 2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])

            Aa = Aa[:, :, :, 1]
            Xa = Xa[:, :, :, 1]

        # 노드가 4, 2, 1개만 남은 경우
        if Xa.size(2) == 4:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            Aa[:, :, 1].mul_(Aa[:, :, 0])

            Xa[:, :, 3].add_(Aa[:, :, 3].mul(Xa[:, :, 2] + Aa[:, :, 2].mul(Xa[:, :, 1])))
        elif Xa.size(2) == 2:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            return
        else:
            return

        # down sweep (처음 2단계는 풀어서 처리)
        Aa = A[:, :, 2 ** (num_steps - 2) - 1 : L : 2 ** (num_steps - 2)]
        Xa = X[:, :, 2 ** (num_steps - 2) - 1 : L : 2 ** (num_steps - 2)]
        Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 1]))
        Aa[:, :, 2].mul_(Aa[:, :, 1])

        for k in range(num_steps - 3, -1, -1):
            Aa = A[:, :, 2**k - 1 : L : 2**k]
            Xa = X[:, :, 2**k - 1 : L : 2**k]

            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
            Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])

    @staticmethod
    def pscan_rev(A, X):
        # 위와 동일하지만 역방향(backward pass에서 사용).
        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        Aa = A
        Xa = X
        for _ in range(num_steps - 2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, :, 0].add_(Aa[:, :, :, 0].mul(Xa[:, :, :, 1]))
            Aa[:, :, :, 0].mul_(Aa[:, :, :, 1])

            Aa = Aa[:, :, :, 0]
            Xa = Xa[:, :, :, 0]

        if Xa.size(2) == 4:
            Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 3]))
            Aa[:, :, 2].mul_(Aa[:, :, 3])

            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1].add(Aa[:, :, 1].mul(Xa[:, :, 2]))))
        elif Xa.size(2) == 2:
            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1]))
            return
        else:
            return

        Aa = A[:, :, 0:L : 2 ** (num_steps - 2)]
        Xa = X[:, :, 0:L : 2 ** (num_steps - 2)]
        Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 2]))
        Aa[:, :, 1].mul_(Aa[:, :, 2])

        for k in range(num_steps - 3, -1, -1):
            Aa = A[:, :, 0:L:2**k]
            Xa = X[:, :, 0:L:2**k]

            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, :-1, 1].add_(Aa[:, :, :-1, 1].mul(Xa[:, :, 1:, 0]))
            Aa[:, :, :-1, 1].mul_(Aa[:, :, 1:, 0])

    @staticmethod
    def forward(ctx, A_in, X_in):
        """병렬 스캔. 가능하면 2의 거듭제곱 길이를 쓰는 게 유리하다.

        Args:
            A_in : (B, L, D, N)
            X_in : (B, L, D, N)

        Returns:
            H : (B, L, D, N)
        """
        L = X_in.size(1)

        # in-place 연산 때문에 clone이 필요하다.
        if L == npo2(L):
            A = A_in.clone()
            X = X_in.clone()
        else:
            A = pad_npo2(A_in)  # (B, npo2(L), D, N)
            X = pad_npo2(X_in)  # (B, npo2(L), D, N)

        A = A.transpose(2, 1)  # (B, D, npo2(L), N)
        X = X.transpose(2, 1)  # (B, D, npo2(L), N)

        PScan.pscan(A, X)

        ctx.save_for_backward(A_in, X)

        return X.transpose(2, 1)[:, :L]

    @staticmethod
    def backward(ctx, grad_output_in):
        """출력에서 입력으로 그래디언트를 흘린다.

        Args:
            ctx : A_in : (B, L, D, N), X : (B, D, L, N)
            grad_output_in : (B, L, D, N)

        Returns:
            gradA : (B, L, D, N), gradX : (B, L, D, N)
        """
        A_in, X = ctx.saved_tensors

        L = grad_output_in.size(1)

        if L == npo2(L):
            grad_output = grad_output_in.clone()
        else:
            grad_output = pad_npo2(grad_output_in)  # (B, npo2(L), D, N)
            A_in = pad_npo2(A_in)  # (B, npo2(L), D, N)

        grad_output = grad_output.transpose(2, 1)
        A_in = A_in.transpose(2, 1)  # (B, D, npo2(L), N)
        A = torch.nn.functional.pad(A_in[:, :, 1:], (0, 0, 0, 1))  # 1칸 왼쪽으로 shift

        PScan.pscan_rev(A, grad_output)

        Q = torch.zeros_like(X)
        Q[:, :, 1:].add_(X[:, :, :-1] * grad_output[:, :, 1:])

        return Q.transpose(2, 1)[:, :L], grad_output.transpose(2, 1)[:, :L]


pscan = PScan.apply
