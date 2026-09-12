"""Train the 768 -> H -> 1 NNUE on packed Lichess/Stockfish evaluations and quantise it.

Training tooling only (torch); nothing here runs on the competition platform.  The first layer is
an ``EmbeddingBag`` sum over the at-most-32 active features, which is exactly the accumulator the
numba inference recomputes, so float and int paths share one definition of the network.

Target: ``sigmoid(cp / CP_SCALE)`` win probability, loss MSE in probability space.  The network's
raw output is therefore a logit and ``output * CP_SCALE`` is a centipawn score.

The first-layer weights are clipped after every step so that the worst-case int16 accumulator
(bias plus the largest same-sign weights any legal piece placement can select) cannot overflow;
``--quantise-only`` re-runs the quantisation and the held-out report from a saved checkpoint.
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
from encoding import KING_BUCKET, MAX_PIECES, NUM_FEATURES

import fastboard as fb
from nnue_eval import BUCKET_DIVISOR, CP_SCALE, QA, QB

QB2 = 256  # scale of the second layer's weights (int16); its sums are formed in int64

torch.set_num_threads(4)


@numba.njit(cache=True)
def decode_batch(
    occ: np.ndarray, nib: np.ndarray, rows: np.ndarray, king_buckets: int
) -> tuple[np.ndarray, np.ndarray]:
    """Flat active-feature indices plus per-sample offsets, for torch's EmbeddingBag.

    With ``king_buckets`` > 1 the board is mirrored so the side to move's king (code 5) sits on
    files a-d and every feature is offset by that king square's bucket (see encoding.KING_BUCKET).
    """
    n = rows.shape[0]
    flat = np.empty(n * MAX_PIECES, dtype=np.int64)
    offsets = np.empty(n, dtype=np.int64)
    one = np.uint64(1)
    pos = 0
    for i in range(n):
        row = rows[i]
        offsets[i] = pos
        mirror = np.int64(0)
        offset = np.int64(0)
        if king_buckets > 1:
            bits = occ[row]
            count = 0
            king = np.int64(0)
            while bits:
                square = np.int64(fb.lsb_square(bits))
                bits &= bits - one
                byte = np.int64(nib[row, count >> 1])
                shift = 4 if count & 1 else 0
                if (byte >> shift) & 15 == 5:
                    king = square
                count += 1
            if (king & 7) >= 4:
                mirror = np.int64(7)
            offset = KING_BUCKET[king ^ mirror] * NUM_FEATURES
        bits = occ[row]
        count = 0
        while bits:
            square = np.int64(fb.lsb_square(bits))
            bits &= bits - one
            byte = np.int64(nib[row, count >> 1])
            shift = 4 if count & 1 else 0
            code = (byte >> shift) & 15
            flat[pos] = offset + code * 64 + (square ^ mirror)
            pos += 1
            count += 1
    return flat[:pos], offsets


class Nnue(nn.Module):
    """768 -> hidden (clipped ReLU) -> 1, defined once for both the float and the int path.
    With ``king_buckets`` > 1 the input table has 768 rows per king bucket."""

    def __init__(
        self, hidden: int, king_buckets: int = 1, layer2: int = 0, factoriser: bool = False,
        buckets: int = 1, screlu: bool = False,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.king_buckets = king_buckets
        self.layer2 = layer2
        self.embed = nn.EmbeddingBag(NUM_FEATURES * king_buckets, hidden, mode="sum")
        # factoriser: a shared 768-row table added to every king bucket during training and
        # merged into the bucketed table at export, so rare buckets learn from all positions
        self.factor = (
            nn.EmbeddingBag(NUM_FEATURES, hidden, mode="sum")
            if factoriser and king_buckets > 1
            else None
        )
        if self.factor is not None:
            nn.init.zeros_(self.factor.weight)
        self.bias1 = nn.Parameter(torch.zeros(hidden))
        # optional second clipped-ReLU layer (hidden -> layer2) before the output
        self.mid = nn.Linear(hidden, layer2) if layer2 else None
        self.buckets = buckets
        self.screlu = screlu
        self.out = nn.Linear(layer2 if layer2 else hidden, buckets)
        nn.init.normal_(self.embed.weight, std=0.05)
        nn.init.normal_(self.out.weight, std=0.1)
        nn.init.zeros_(self.out.bias)

    def merged_input_weight(self) -> torch.Tensor:
        """The input table the engine sees: bucketed weights plus the shared factoriser."""
        if self.factor is None:
            return self.embed.weight
        stacked = self.embed.weight.view(self.king_buckets, NUM_FEATURES, self.hidden)
        return (stacked + self.factor.weight.unsqueeze(0)).reshape(-1, self.hidden)

    def clip_(self, clip: float) -> None:
        """Keep every merged input weight within +/-clip (the int16 bound is checked at export)."""
        self.bias1.clamp_(-clip, clip)
        if self.factor is None:
            self.embed.weight.clamp_(-clip, clip)
            return
        self.factor.weight.clamp_(-clip, clip)
        shared = self.factor.weight.unsqueeze(0)
        stacked = self.embed.weight.view(self.king_buckets, NUM_FEATURES, self.hidden)
        stacked.clamp_(min=-clip - shared, max=clip - shared)

    def forward(self, flat: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        accumulator = self.embed(flat, offsets) + self.bias1
        if self.factor is not None:
            accumulator = accumulator + self.factor(flat % NUM_FEATURES, offsets)
        hidden = torch.clamp(accumulator, 0.0, 1.0)
        if self.screlu:
            hidden = hidden * hidden
        if self.mid is not None:
            hidden = torch.clamp(self.mid(hidden), 0.0, 1.0)
        out = self.out(hidden)
        if self.buckets == 1:
            return out.squeeze(1)
        pieces = torch.diff(offsets, append=torch.tensor([flat.shape[0]], device=offsets.device))
        index = torch.clamp((pieces - 2) // BUCKET_DIVISOR, 0, self.buckets - 1)
        return out.gather(1, index.unsqueeze(1).long()).squeeze(1)


# Per side: one king plus at most 15 other pieces, with at most 8 pawns (never on ranks 1 or 8),
# 10 knights, 10 bishops, 10 rooks or 9 queens.  Promotions cannot exceed these caps.
PIECE_CAPS = {0: 8, 1: 10, 2: 10, 3: 10, 4: 9}


def _legal_side_sum(weight: np.ndarray, sign: int) -> np.ndarray:
    """Largest same-sign contribution of both sides' pieces under the legal piece caps.

    Only the positive (``sign`` = +1) or negative (-1) parts of the weights are summed, so the
    result also bounds every partial sum the engine forms while recomputing the accumulator.
    """
    part = np.clip(sign * weight, 0, None)
    total = np.zeros(weight.shape[1], dtype=np.int64)
    for colour in (0, 6):
        king = part[(colour + 5) * 64:(colour + 6) * 64].max(axis=0)
        pool = []
        for piece, cap in PIECE_CAPS.items():
            block = part[(colour + piece) * 64:(colour + piece + 1) * 64].copy()
            if piece == 0:
                block[:8] = 0
                block[56:] = 0
            pool.append(np.sort(block, axis=0)[-cap:])
        total += king + np.sort(np.concatenate(pool, axis=0), axis=0)[-15:].sum(axis=0)
    return total


def accumulator_bound(weight: np.ndarray, bias: np.ndarray) -> int:
    """Worst-case |int16 accumulator| over any legal placement of pieces (and any partial sum).
    A position uses one king bucket, so the bound is the worst over the 768-row blocks."""
    bound = 0
    for block in weight.reshape(-1, NUM_FEATURES, weight.shape[1]):
        high = np.abs(bias + _legal_side_sum(block, 1)).max()
        low = np.abs(bias - _legal_side_sum(block, -1)).max()
        bound = max(bound, int(high), int(low))
    return bound


def accumulator_bound_any32(weight: np.ndarray, bias: np.ndarray) -> int:
    """The older, looser bound: bias plus the 32 largest same-sign weights of any piece type."""
    bound = 0
    for block in weight.reshape(-1, NUM_FEATURES, weight.shape[1]):
        ordered = np.sort(block, axis=0)
        high = np.abs(bias + ordered[-MAX_PIECES:].sum(axis=0)).max()
        low = np.abs(bias + ordered[:MAX_PIECES].sum(axis=0)).max()
        bound = max(bound, int(high), int(low))
    return bound


def quantise(model: Nnue) -> dict[str, np.ndarray]:
    weight1 = model.merged_input_weight().detach().cpu().numpy().astype(np.float64)
    bias1 = model.bias1.detach().cpu().numpy().astype(np.float64)
    weight2 = model.out.weight.detach().cpu().numpy().astype(np.float64)  # (buckets, hidden)
    bias2 = model.out.bias.detach().cpu().numpy().astype(np.float64)  # (buckets,)
    w1 = np.rint(weight1 * QA).astype(np.int16)
    b1 = np.rint(bias1 * QA).astype(np.int16)
    if model.mid is not None:
        # two layers: w2/b2 are the middle layer (hidden -> layer2, scale QB2), w3/b3 the output
        mid_w = model.mid.weight.detach().cpu().numpy().astype(np.float64)
        mid_b = model.mid.bias.detach().cpu().numpy().astype(np.float64)
        assert np.abs(mid_w).max() * QB2 < 32767, "layer-2 weight exceeds int16 at QB2"
        return {
            "w1": w1,
            "b1": b1,
            "w2": np.rint(mid_w * QB2).astype(np.int16),
            "b2": np.rint(mid_b * QA * QB2).astype(np.int32),
            "w3": np.rint(weight2 * QB).astype(np.int16),
            "b3": np.int32(round(bias2 * QA * QB)),
        }
    w2 = np.rint(weight2 * QB).astype(np.int16)
    b2 = np.rint(bias2 * QA * QB).astype(np.int32)
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def quantised_forward(weights: dict[str, np.ndarray], flat: np.ndarray, offsets: np.ndarray,
                      count: int, screlu: bool = False) -> np.ndarray:
    """Integer forward pass in numpy, bit-identical to the numba inference."""
    w1, b1, w2, b2 = weights["w1"], weights["b1"], weights["w2"], weights["b2"]
    two_layer = "w3" in weights
    w2r = w2 if w2.ndim == 2 else w2.reshape(1, -1)
    b2r = np.atleast_1d(b2)
    buckets = w2r.shape[0]
    out = np.empty(count, dtype=np.int64)
    ends = np.append(offsets[1:], len(flat))
    for i in range(count):
        acc = b1.astype(np.int32) + w1[flat[offsets[i] : ends[i]]].sum(axis=0, dtype=np.int32)
        acc = np.clip(acc, 0, QA).astype(np.int64)
        if two_layer:
            mid = b2.astype(np.int64) + w2.astype(np.int64) @ acc
            mid = np.clip(mid // QB2, 0, QA)
            out[i] = int(weights["b3"]) + int((mid * weights["w3"].astype(np.int64)).sum())
            continue
        if screlu:
            acc = acc * acc // QA
        bucket = 0
        if buckets > 1:
            pieces = int(ends[i] - offsets[i])
            bucket = int(np.clip((pieces - 2) // BUCKET_DIVISOR, 0, buckets - 1))
        out[i] = int(b2r[bucket]) + int((acc * w2r[bucket].astype(np.int64)).sum())
    return out * CP_SCALE // (QA * QB)


def report(model: Nnue, weights: dict[str, np.ndarray], occ: np.ndarray, nib: np.ndarray,
           scores: np.ndarray, rows: np.ndarray, quant_sample: int,
           king_buckets: int = 1) -> dict[str, float]:
    """Held-out centipawn MAE / correlation for the float net and for the quantised net."""
    model.eval()
    device = next(model.parameters()).device
    predictions = np.empty(len(rows), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(rows), 65536):
            chunk = rows[start : start + 65536]
            flat, offsets = decode_batch(occ, nib, chunk, king_buckets)
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
    flat, offsets = decode_batch(occ, nib, sample, king_buckets)
    quant_cp = quantised_forward(
        weights, flat, offsets, len(sample), model.screlu
    ).astype(np.float32)
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
    parser.add_argument("--data", type=Path, default=Path("training/data/positions.npy"))
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--holdout", type=int, default=1_000_000)
    parser.add_argument("--train-limit", type=int, default=0, help="0 = all remaining positions")
    parser.add_argument("--quant-sample", type=int, default=100_000)
    parser.add_argument("--clip", type=float, default=1.98, help="first-layer weight clip")
    parser.add_argument("--out", type=Path, default=Path("training/nnue.npz"))
    parser.add_argument("--checkpoint", type=Path, default=Path("training/data/nnue_float.pt"))
    parser.add_argument("--curve", type=Path, default=Path("training/data/curve.json"))
    parser.add_argument("--quantise-only", action="store_true")
    parser.add_argument("--buckets", type=int, default=1,
                        help="output buckets chosen by piece count ((pieces - 2) // 4)")
    parser.add_argument("--screlu", action="store_true",
                        help="square the clipped activation (SCReLU)")
    parser.add_argument("--loss-exp", type=float, default=2.0,
                        help="loss = |sigmoid(v) - t|^exp; 2.5 reported +10..+20 in other engines")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1 (0.95 reported +4)")
    parser.add_argument("--schedule", choices=("step", "cosine"), default="step",
                        help="step: x0.7 per epoch (the old default); cosine: lr -> lr/100")
    parser.add_argument("--factoriser", action="store_true",
                        help="with --king-buckets: train a shared 768-row table on top of the "
                             "bucketed one and merge it at export")
    parser.add_argument(
        "--layer2", type=int, default=0,
        help="0 = single hidden layer; N = add a second clipped-ReLU layer of N units",
    )
    parser.add_argument(
        "--king-buckets", type=int, default=1,
        help="1 = plain 768 inputs; 8 = mirror the board to the king's half and offset the "
             "features by the king's bucket (encoding.KING_BUCKET), an 8 x 768 x hidden table",
    )
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
    model = Nnue(args.hidden, args.king_buckets, args.layer2, args.factoriser,
                 args.buckets, args.screlu)
    king_buckets = args.king_buckets
    if args.quantise_only:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.to(device)
    if not args.quantise_only:
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
        if args.schedule == "cosine":
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser, T_max=args.epochs, eta_min=args.lr * 0.01
            )
        else:
            schedule = torch.optim.lr_scheduler.StepLR(optimiser, step_size=1, gamma=0.7)
        targets = torch.sigmoid(torch.from_numpy(scores.astype(np.float32)) / CP_SCALE)
        if results is not None:
            # result -2 marks rows without a game result (e.g. Lichess evaluations): eval only
            has_result = torch.from_numpy(results != -2)
            lam = torch.where(has_result, torch.tensor(args.wdl), torch.tensor(0.0))
            wdl = torch.from_numpy(np.clip((results + 1.0) / 2.0, 0.0, 1.0).astype(np.float32))
            targets = (1.0 - lam) * targets + lam * wdl
            print(
                f"target: {1 - args.wdl:.2f} * eval + {args.wdl:.2f} * result on "
                f"{int(has_result.sum()):,} rows with a result, eval only on the rest",
                flush=True,
            )
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
                flat, offsets = decode_batch(occ, nib, chunk, king_buckets)
                value = model(
                    torch.from_numpy(flat).to(device), torch.from_numpy(offsets).to(device)
                )
                target = targets[torch.from_numpy(chunk).to(device)]
                loss = torch.mean(torch.abs(torch.sigmoid(value) - target) ** args.loss_exp)
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()
                with torch.no_grad():
                    model.clip_(args.clip)
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
            metrics = report(model, weights, occ, nib, scores, holdout[:200_000], 20_000,
                             king_buckets)
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
    w1_wide, b1_wide = weights["w1"].astype(np.int64), weights["b1"].astype(np.int64)
    bound = accumulator_bound(w1_wide, b1_wide)
    print(f"worst-case |accumulator| = {bound} over legal placements (int16 limit 32767; "
          f"any-32-pieces bound {accumulator_bound_any32(w1_wide, b1_wide)})")
    if bound > 32767:
        raise SystemExit("accumulator would overflow int16: lower --clip and retrain")
    np.savez_compressed(args.out, **weights)
    metrics = report(model, weights, occ, nib, scores, holdout, args.quant_sample, king_buckets)
    print(json.dumps(metrics, indent=2))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
