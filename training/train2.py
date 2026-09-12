"""Train the second-generation net: (768 -> H)x2 -> 8 buckets -> 1, SCReLU, and quantise it.

    uv run python training/train2.py --data training/data/positions_gen345.npy --hidden 384 ...

Two perspectives share one first layer: the side to move sees the canonical features the records
store; the opponent sees the same position with the roles swapped and the board flipped
(feature f = code * 64 + square -> ((code + 6) % 12) * 64 + (square ^ 56)). Each accumulator goes
through SCReLU (clamp to [0, 1], then square), the two are concatenated, and one of eight output
rows is chosen by the number of pieces on the board. Integer inference: int16 first layer
scaled by QA, int32 squares (<= QA^2), int64 dot with the int16 output row scaled by QB, divided
by QA, plus the bucket bias scaled by QA*QB; ``quantised_forward2`` here is the bit-exact
reference the numba kernel is tested against. Targets and reporting follow train.py.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import _paths  # noqa: F401  (training/ and engine/ on sys.path)
from encoding import NUM_FEATURES
from nnue_eval import CP_SCALE, QA, QB
from train import accumulator_bound, decode_batch

BUCKETS = 8


def bucket_of(piece_count: np.ndarray) -> np.ndarray:
    """0..7 by piece count: 2-5, 6-9, ..., 30-32 (kings included)."""
    return np.minimum((piece_count - 2) // 4, BUCKETS - 1).astype(np.int64)


def opponent_view(flat: np.ndarray) -> np.ndarray:
    code = flat // 64
    square = flat & 63
    return ((code + 6) % 12) * 64 + (square ^ 56)


def decode2(occ: np.ndarray, nib: np.ndarray, rows: np.ndarray):
    flat, offsets = decode_batch(occ, nib, rows)
    counts = np.diff(np.append(offsets, len(flat)))
    return flat, opponent_view(flat), offsets, bucket_of(counts)


class Nnue2(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.embed = nn.EmbeddingBag(NUM_FEATURES, hidden, mode="sum")
        self.bias1 = nn.Parameter(torch.zeros(hidden))
        self.out = nn.Linear(2 * hidden, BUCKETS)
        nn.init.normal_(self.embed.weight, std=0.05)
        nn.init.normal_(self.out.weight, std=0.05)
        nn.init.zeros_(self.out.bias)

    def forward(self, flat_stm, flat_nstm, offsets, bucket):
        us = torch.clamp(self.embed(flat_stm, offsets) + self.bias1, 0.0, 1.0)
        them = torch.clamp(self.embed(flat_nstm, offsets) + self.bias1, 0.0, 1.0)
        hidden = torch.cat([us * us, them * them], dim=1)
        return self.out(hidden).gather(1, bucket[:, None]).squeeze(1)


def quantise2(model: Nnue2) -> dict[str, np.ndarray]:
    w1 = model.embed.weight.detach().cpu().numpy().astype(np.float64)
    b1 = model.bias1.detach().cpu().numpy().astype(np.float64)
    w2 = model.out.weight.detach().cpu().numpy().astype(np.float64)  # (BUCKETS, 2H)
    b2 = model.out.bias.detach().cpu().numpy().astype(np.float64)
    return {
        "w1": np.rint(w1 * QA).astype(np.int16),
        "b1": np.rint(b1 * QA).astype(np.int16),
        "w2": np.rint(w2 * QB).astype(np.int16),
        "b2": np.rint(b2 * QA * QB).astype(np.int32),
        "arch": np.array("dual-screlu-buckets8"),
    }


def quantised_forward2(weights, flat_stm, flat_nstm, offsets, bucket, count) -> np.ndarray:
    """Integer forward pass, the reference for the numba kernel."""
    w1, b1, w2, b2 = weights["w1"], weights["b1"], weights["w2"], weights["b2"]
    ends = np.append(offsets[1:], len(flat_stm))
    out = np.empty(count, dtype=np.int64)
    for i in range(count):
        a = b1.astype(np.int32) + w1[flat_stm[offsets[i] : ends[i]]].sum(axis=0, dtype=np.int32)
        t = b1.astype(np.int32) + w1[flat_nstm[offsets[i] : ends[i]]].sum(axis=0, dtype=np.int32)
        a = np.clip(a, 0, QA).astype(np.int64)
        t = np.clip(t, 0, QA).astype(np.int64)
        hidden = np.concatenate([a * a, t * t])  # <= QA^2 each
        dot = int((hidden * w2[bucket[i]].astype(np.int64)).sum())
        out[i] = dot // QA + int(b2[bucket[i]])
    return out * CP_SCALE // (QA * QB)


def report2(model, weights, occ, nib, scores, rows, quant_sample, device) -> dict[str, float]:
    model.eval()
    predictions = np.empty(len(rows), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(rows), 65536):
            chunk = rows[start : start + 65536]
            f, g, o, b = decode2(occ, nib, chunk)
            value = model(
                torch.from_numpy(f).to(device), torch.from_numpy(g).to(device),
                torch.from_numpy(o).to(device), torch.from_numpy(b).to(device),
            )
            predictions[start : start + len(chunk)] = value.cpu().numpy()
    target = scores[rows].astype(np.float32)
    float_cp = np.clip(predictions * CP_SCALE, -2000, 2000)
    metrics = {
        "float_mae_cp": float(np.abs(float_cp - target).mean()),
        "float_corr": float(np.corrcoef(float_cp, target)[0, 1]),
    }
    sample = rows[:quant_sample]
    f, g, o, b = decode2(occ, nib, sample)
    quant_cp = np.clip(quantised_forward2(weights, f, g, o, b, len(sample)), -2000, 2000)
    metrics["quant_mae_cp"] = float(np.abs(quant_cp - scores[sample]).mean())
    metrics["quant_corr"] = float(np.corrcoef(quant_cp, scores[sample].astype(np.float32))[0, 1])
    metrics["quant_vs_float_mae_cp"] = float(np.abs(quant_cp - float_cp[:quant_sample]).mean())
    metrics["quant_vs_float_max_cp"] = float(np.abs(quant_cp - float_cp[:quant_sample]).max())
    quiet = np.abs(target) < 400
    metrics["float_mae_cp_under400"] = float(np.abs(float_cp[quiet] - target[quiet]).mean())
    metrics["float_sign_agreement"] = float(np.mean((float_cp > 0) == (target > 0)))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--holdout", type=int, default=400_000)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--quant-sample", type=int, default=100_000)
    parser.add_argument("--clip", type=float, default=1.98)
    parser.add_argument("--wdl", type=float, default=0.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--curve", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    device = torch.device(args.device)
    print(f"{total:,} positions: {len(train_rows):,} train / {len(holdout):,} held out; {device}")

    model = Nnue2(args.hidden).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    schedule = torch.optim.lr_scheduler.StepLR(optimiser, step_size=1, gamma=0.7)
    targets = torch.sigmoid(torch.from_numpy(scores.astype(np.float32)) / CP_SCALE)
    if results is not None:
        targets = (1.0 - args.wdl) * targets + args.wdl * torch.from_numpy((results + 1.0) / 2.0)
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
            f, g, o, b = decode2(occ, nib, chunk)
            value = model(
                torch.from_numpy(f).to(device), torch.from_numpy(g).to(device),
                torch.from_numpy(o).to(device), torch.from_numpy(b).to(device),
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
                rate = steps * args.batch / (time.perf_counter() - started)
                print(
                    f"  epoch {epoch} step {steps} loss {running / steps:.6f} ({rate:,.0f} pos/s)",
                    flush=True,
                )
        schedule.step()
        weights = quantise2(model)
        metrics = report2(model, weights, occ, nib, scores, holdout[:200_000], 20_000, device)
        entry = {"epoch": epoch, "train_loss": running / max(1, steps),
                 "seconds": time.perf_counter() - started, **metrics}
        curve.append(entry)
        print(json.dumps(entry), flush=True)
        torch.save(model.state_dict(), args.checkpoint)
        args.curve.write_text(json.dumps(curve, indent=2))

    weights = quantise2(model)
    bound = accumulator_bound(weights["w1"].astype(np.int32), weights["b1"].astype(np.int32))
    print(f"worst-case |accumulator| = {bound} (int16 limit 32767)")
    if bound > 32767:
        raise SystemExit("accumulator would overflow int16: lower --clip and retrain")
    np.savez_compressed(args.out, **weights)
    metrics = report2(model, weights, occ, nib, scores, holdout, args.quant_sample, device)
    print(json.dumps(metrics, indent=2))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
