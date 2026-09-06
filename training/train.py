"""Train the 768 -> H -> 1 NNUE on packed Lichess/Stockfish evaluations and quantise it.

Training tooling only (torch); nothing here runs on the competition platform.  The first layer is
an ``EmbeddingBag`` sum over the at-most-32 active features, which is exactly the accumulator the
numba inference recomputes, so float and int paths share one definition of the network.

Target: ``sigmoid(cp / CP_SCALE)`` win probability, loss MSE in probability space.  The network's
raw output is therefore a logit and ``output * CP_SCALE`` is a centipawn score.

The first-layer weights are clipped after every step so that the worst-case int16 accumulator
(bias plus the 32 largest same-sign weights of a column) cannot overflow; ``--quantise-only``
re-runs the quantisation and the held-out report from a saved checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numba
from encoding import MAX_PIECES, NUM_FEATURES

import fastboard as fb
from nnue_eval import CP_SCALE, QA, QB

torch.set_num_threads(4)


@numba.njit(cache=True)
def decode_batch(
    occ: np.ndarray, nib: np.ndarray, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Flat active-feature indices plus per-sample offsets, for torch's EmbeddingBag."""
    n = rows.shape[0]
    flat = np.empty(n * MAX_PIECES, dtype=np.int64)
    offsets = np.empty(n, dtype=np.int64)
    one = np.uint64(1)
    pos = 0
    for i in range(n):
        row = rows[i]
        offsets[i] = pos
        bits = occ[row]
        count = 0
        while bits:
            square = fb.lsb_square(bits)
            bits &= bits - one
            byte = np.int64(nib[row, count >> 1])
            shift = 4 if count & 1 else 0
            code = (byte >> shift) & 15
            flat[pos] = code * 64 + square
            pos += 1
            count += 1
    return flat[:pos], offsets


class Nnue(nn.Module):
    """768 -> hidden (clipped ReLU) -> 1, defined once for both the float and the int path."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.embed = nn.EmbeddingBag(NUM_FEATURES, hidden, mode="sum")
        self.bias1 = nn.Parameter(torch.zeros(hidden))
        self.out = nn.Linear(hidden, 1)
        nn.init.normal_(self.embed.weight, std=0.05)
        nn.init.normal_(self.out.weight, std=0.1)
        nn.init.zeros_(self.out.bias)

    def forward(self, flat: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        accumulator = self.embed(flat, offsets) + self.bias1
        return self.out(torch.clamp(accumulator, 0.0, 1.0)).squeeze(1)


def accumulator_bound(weight: np.ndarray, bias: np.ndarray) -> int:
    """Worst-case |int16 accumulator| over any placement of at most 32 pieces."""
    ordered = np.sort(weight, axis=0)
    high = np.abs(bias + ordered[-MAX_PIECES:].sum(axis=0)).max()
    low = np.abs(bias + ordered[:MAX_PIECES].sum(axis=0)).max()
    return int(max(high, low))


def quantise(model: Nnue) -> dict[str, np.ndarray]:
    weight1 = model.embed.weight.detach().cpu().numpy().astype(np.float64)
    bias1 = model.bias1.detach().cpu().numpy().astype(np.float64)
    weight2 = model.out.weight.detach().cpu().numpy().reshape(-1).astype(np.float64)
    bias2 = float(model.out.bias.detach().cpu().numpy()[0])
    w1 = np.rint(weight1 * QA).astype(np.int16)
    b1 = np.rint(bias1 * QA).astype(np.int16)
    w2 = np.rint(weight2 * QB).astype(np.int16)
    b2 = np.int32(round(bias2 * QA * QB))
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def quantised_forward(weights: dict[str, np.ndarray], flat: np.ndarray, offsets: np.ndarray,
                      count: int) -> np.ndarray:
    """Integer forward pass in numpy, bit-identical to the numba inference."""
    w1, b1, w2, b2 = weights["w1"], weights["b1"], weights["w2"], weights["b2"]
    out = np.empty(count, dtype=np.int64)
    ends = np.append(offsets[1:], len(flat))
    for i in range(count):
        acc = b1.astype(np.int32) + w1[flat[offsets[i] : ends[i]]].sum(axis=0, dtype=np.int32)
        acc = np.clip(acc, 0, QA).astype(np.int16)
        out[i] = int(b2) + int((acc.astype(np.int32) * w2.astype(np.int32)).sum())
    return out * CP_SCALE // (QA * QB)


def report(model: Nnue, weights: dict[str, np.ndarray], occ: np.ndarray, nib: np.ndarray,
           scores: np.ndarray, rows: np.ndarray, quant_sample: int) -> dict[str, float]:
    """Held-out centipawn MAE / correlation for the float net and for the quantised net."""
    model.eval()
    device = next(model.parameters()).device
    predictions = np.empty(len(rows), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(rows), 65536):
            chunk = rows[start : start + 65536]
            flat, offsets = decode_batch(occ, nib, chunk)
            value = model(torch.from_numpy(flat).to(device), torch.from_numpy(offsets).to(device))
            predictions[start : start + len(chunk)] = value.cpu().numpy()
    target = scores[rows].astype(np.float32)
    float_cp = np.clip(predictions * CP_SCALE, -2000, 2000)
    metrics = {
        "float_mae_cp": float(np.abs(float_cp - target).mean()),
        "float_corr": float(np.corrcoef(float_cp, target)[0, 1]),
        "float_prob_mse": float(
            ((1 / (1 + np.exp(-predictions)) - 1 / (1 + np.exp(-target / CP_SCALE))) ** 2).mean()
        ),
    }
    sample = rows[:quant_sample]
    flat, offsets = decode_batch(occ, nib, sample)
    quant_cp = quantised_forward(weights, flat, offsets, len(sample)).astype(np.float32)
    quant_cp = np.clip(quant_cp, -2000, 2000)
    metrics["quant_mae_cp"] = float(np.abs(quant_cp - scores[sample]).mean())
    metrics["quant_corr"] = float(np.corrcoef(quant_cp, scores[sample].astype(np.float32))[0, 1])
    metrics["quant_vs_float_mae_cp"] = float(np.abs(quant_cp - float_cp[:quant_sample]).mean())
    metrics["quant_vs_float_max_cp"] = float(np.abs(quant_cp - float_cp[:quant_sample]).max())
    # The +/-2000 mate clamp is 13% of the data and a static evaluation cannot predict it, so a
    # second MAE over the quiet band says more about ordinary play.
    quiet = np.abs(target) < 400
    metrics["float_mae_cp_under400"] = float(np.abs(float_cp[quiet] - target[quiet]).mean())
    metrics["float_sign_agreement"] = float(np.mean((float_cp > 0) == (target > 0)))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("work/nnue/data/positions.npy"))
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--holdout", type=int, default=1_000_000)
    parser.add_argument("--train-limit", type=int, default=0, help="0 = all remaining positions")
    parser.add_argument("--quant-sample", type=int, default=100_000)
    parser.add_argument("--clip", type=float, default=1.98, help="first-layer weight clip")
    parser.add_argument("--out", type=Path, default=Path("work/nnue/nnue.npz"))
    parser.add_argument("--checkpoint", type=Path, default=Path("work/nnue/data/nnue_float.pt"))
    parser.add_argument("--curve", type=Path, default=Path("work/nnue/data/curve.json"))
    parser.add_argument("--quantise-only", action="store_true")
    parser.add_argument(
        "--wdl",
        type=float,
        default=0.0,
        help="weight of the game result in the target: (1-wdl)*sigmoid(score/CP_SCALE) + "
        "wdl*(result+1)/2; needs a set built with build_set.py --with-result",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device for the float training (the data pipeline stays on the CPU)",
    )
    args = parser.parse_args()

    records = np.load(args.data, mmap_mode="r")
    occ = np.ascontiguousarray(records["occ"])
    nib = np.ascontiguousarray(records["nib"])
    scores = np.ascontiguousarray(records["score"]).astype(np.int16)
    results = None
    if args.wdl > 0:
        if "result" not in (records.dtype.names or ()):
            raise SystemExit("--wdl needs a set built with build_set.py --with-result")
        results = np.ascontiguousarray(records["result"]).astype(np.float32)
    del records
    total = len(occ)
    rng = np.random.default_rng(20260904)
    order = rng.permutation(total)
    holdout = order[: args.holdout]
    train_rows = order[args.holdout :]
    if args.train_limit:
        train_rows = train_rows[: args.train_limit]
    print(f"{total:,} positions: {len(train_rows):,} train / {len(holdout):,} held out")

    device = torch.device(args.device)
    print(f"device {device}", flush=True)
    model = Nnue(args.hidden)
    if args.quantise_only:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.to(device)
    if not args.quantise_only:
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
        schedule = torch.optim.lr_scheduler.StepLR(optimiser, step_size=1, gamma=0.7)
        targets = torch.sigmoid(torch.from_numpy(scores.astype(np.float32)) / CP_SCALE)
        if results is not None:
            wdl = torch.from_numpy((results + 1.0) / 2.0)
            targets = (1.0 - args.wdl) * targets + args.wdl * wdl
            print(f"target: {1 - args.wdl:.2f} * eval + {args.wdl:.2f} * result", flush=True)
        targets = targets.to(device)
        curve = []
        for epoch in range(args.epochs):
            model.train()
            epoch_rows = train_rows[rng.permutation(len(train_rows))]
            started = time.perf_counter()
            running = 0.0
            steps = 0
            for start in range(0, len(epoch_rows) - args.batch + 1, args.batch):
                chunk = epoch_rows[start : start + args.batch]
                flat, offsets = decode_batch(occ, nib, chunk)
                value = model(
                    torch.from_numpy(flat).to(device), torch.from_numpy(offsets).to(device)
                )
                target = targets[torch.from_numpy(chunk).to(device)]
                loss = torch.mean((torch.sigmoid(value) - target) ** 2)
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()
                with torch.no_grad():
                    model.embed.weight.clamp_(-args.clip, args.clip)
                    model.bias1.clamp_(-args.clip, args.clip)
                running += float(loss.detach())
                steps += 1
                if steps % 200 == 0:
                    print(
                        f"  epoch {epoch} step {steps} loss {running / steps:.6f} "
                        f"({steps * args.batch / (time.perf_counter() - started):,.0f} pos/s)",
                        flush=True,
                    )
            schedule.step()
            weights = quantise(model)
            metrics = report(model, weights, occ, nib, scores, holdout[:200_000], 20_000)
            entry = {
                "epoch": epoch,
                "train_loss": running / max(1, steps),
                "seconds": time.perf_counter() - started,
                **metrics,
            }
            curve.append(entry)
            print(json.dumps(entry), flush=True)
            torch.save(model.state_dict(), args.checkpoint)
            args.curve.write_text(json.dumps(curve, indent=2))

    weights = quantise(model)
    bound = accumulator_bound(weights["w1"].astype(np.int32), weights["b1"].astype(np.int32))
    print(f"worst-case |accumulator| = {bound} (int16 limit 32767)")
    if bound > 32767:
        raise SystemExit("accumulator would overflow int16: lower --clip and retrain")
    np.savez_compressed(args.out, **weights)
    metrics = report(model, weights, occ, nib, scores, holdout, args.quant_sample)
    print(json.dumps(metrics, indent=2))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
