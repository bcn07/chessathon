"""Train the dual-perspective NNUE ``(768 * king_buckets -> N) x 2 -> buckets -> 1``.

    uv run python training/train_dual.py --data training/data/positions_bal.npy \
        --hidden 512 --epochs 6 --king-buckets 8 --buckets 8 --clip 1.0 --wdl 0.3 \
        --holdout 500000 --out training/data/nnue_dual_h512.npz

Training tooling only (torch); nothing here runs on the competition platform.  It is ``train.py``
with one change: **two accumulators sharing one input table**.

* P0, the mover's view, is exactly what ``train.py`` builds: canonical codes and squares, the
  left-right mirror and king-bucket offset from the mover's king.
* P1, the opponent's view, is the same function of the colour-swapped, vertically flipped
  position: codes relative to the non-mover, squares ``^ 56``, mirror and bucket from the
  *non-mover's* king.

Both index the same ``EmbeddingBag``, so at N units the input table is half the size of a
single-perspective net at 2N units.  The two SCReLU'd accumulators are concatenated and read
through one of ``--buckets`` output rows chosen by piece count, so ``w2`` is (buckets, 2N).

Two optional heads, both of which change the *shape* of the exported weights and so are recorded
in ``arch`` for the engine's loader to insist on:

``--pairwise`` replaces SCReLU with Stockfish's pairwise multiplication.  Each perspective's N
activations are clipped to [0, 1] and the two halves multiplied elementwise, giving N/2 outputs per
perspective and so **N** columns of ``w2`` instead of 2N.  That is the point of it twice over: the
readout gains genuine cross-terms it could not represent while it was linear in the activations,
and its width halves, so a 2 x 768 net costs the output stage of a single-perspective 768 one.
Quantised, ``y = (clip(a[j]) * clip(a[j + N/2])) // QA`` -- the same [0, QA] range SCReLU's
``v * v // QA`` produced, so the whole fixed-point tail is untouched.

``--psqt`` adds a PSQT skip head: ``wp``, one int32 per input feature per output bucket, summed
over each perspective's active features and added to the positional output as **own minus
opponent**.  It is quantised at ``QA * QB``, the same scale as ``b2``, because it is the same kind
of thing -- a per-feature contribution to the same sum.  The subtraction makes the term exactly
zero in a colour-symmetric position (the two perspectives see the identical feature multiset),
which the engine's ``verify_stack.py`` asserts, and lets the positional half learn a residual
instead of re-deriving material.

Fixed point matches the engine's ``nnue_eval.py`` exactly: ``w1``/``b1`` scaled by QA, the
nonlinearity landing back in [0, QA] on each perspective (``v * v // QA`` for SCReLU,
``a[j] * a[j + N/2] // QA`` for pairwise), ``w2`` scaled by QB, ``b2`` and ``wp`` by QA*QB, and the
centipawn score is ``total * CP_SCALE // (QA * QB)``.  ``quantised_forward_dual`` is the bit-exact
reference the numba kernel is verified against, and it reads the architecture off the weights so it
cannot drift from the file it is handed.  The int16 accumulator bound is
``train.accumulator_bound`` unchanged: each perspective is itself a legal placement over the same
768-row blocks, so the worst case is the same, and neither optional head touches it.
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
from encoding import KING_BUCKET, KING_BUCKET_32, KING_BUCKETS_32, MAX_PIECES, NUM_FEATURES
from train import accumulator_bound, accumulator_bound_any32

import fastboard as fb
from nnue_eval import BUCKET_DIVISOR, CP_SCALE, QA, QB

torch.set_num_threads(4)


@numba.njit(cache=True)
def decode_batch_dual(
    occ: np.ndarray, nib: np.ndarray, rows: np.ndarray, king_buckets: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(P0 indices, P1 indices, per-sample offsets) for torch's EmbeddingBag.

    One pass finds both kings (code 5 is the mover's, code 11 the opponent's), a second emits the
    two feature blocks.  P1's square is P0's square ``^ 56`` and P1's code is P0's code with the
    colour swapped, each mirrored and offset by *its own* king's bucket.
    """
    n = rows.shape[0]
    flat0 = np.empty(n * MAX_PIECES, dtype=np.int64)
    flat1 = np.empty(n * MAX_PIECES, dtype=np.int64)
    offsets = np.empty(n, dtype=np.int64)
    one = np.uint64(1)
    pos = 0
    for i in range(n):
        row = rows[i]
        offsets[i] = pos
        mirror0 = np.int64(0)
        mirror1 = np.int64(0)
        offset0 = np.int64(0)
        offset1 = np.int64(0)
        if king_buckets > 1:
            bits = occ[row]
            count = 0
            king0 = np.int64(0)
            king1 = np.int64(0)
            while bits:
                square = np.int64(fb.lsb_square(bits))
                bits &= bits - one
                byte = np.int64(nib[row, count >> 1])
                shift = 4 if count & 1 else 0
                code = (byte >> shift) & 15
                if code == 5:
                    king0 = square
                elif code == 11:
                    king1 = square ^ np.int64(56)
                count += 1
            if (king0 & 7) >= 4:
                mirror0 = np.int64(7)
            if (king1 & 7) >= 4:
                mirror1 = np.int64(7)
            if king_buckets == KING_BUCKETS_32:
                offset0 = KING_BUCKET_32[king0 ^ mirror0] * NUM_FEATURES
                offset1 = KING_BUCKET_32[king1 ^ mirror1] * NUM_FEATURES
            else:
                offset0 = KING_BUCKET[king0 ^ mirror0] * NUM_FEATURES
                offset1 = KING_BUCKET[king1 ^ mirror1] * NUM_FEATURES
        bits = occ[row]
        count = 0
        while bits:
            square = np.int64(fb.lsb_square(bits))
            bits &= bits - one
            byte = np.int64(nib[row, count >> 1])
            shift = 4 if count & 1 else 0
            code = (byte >> shift) & 15
            other = code - 6 if code >= 6 else code + 6
            flat0[pos] = offset0 + code * 64 + (square ^ mirror0)
            flat1[pos] = offset1 + other * 64 + ((square ^ np.int64(56)) ^ mirror1)
            pos += 1
            count += 1
    return flat0[:pos], flat1[:pos], offsets


class NnueDual(nn.Module):
    """(768 * king_buckets -> hidden) x 2 -> buckets -> 1, one shared input table.

    ``embed`` and ``bias1`` are used for both perspectives; ``out`` reads the mover's half first,
    matching ``nnue_eval.nnue_evaluate_into``.  Its input width is ``2 * hidden`` for SCReLU and
    ``hidden`` for ``--pairwise``, because the pairwise product halves each perspective."""

    def __init__(
        self, hidden: int, king_buckets: int = 1, buckets: int = 1, screlu: bool = True,
        pairwise: bool = False, psqt: bool = False,
    ) -> None:
        super().__init__()
        if pairwise and hidden % 2:
            raise SystemExit(f"--pairwise needs an even --hidden, got {hidden}")
        self.hidden = hidden
        self.king_buckets = king_buckets
        self.buckets = buckets
        self.screlu = screlu
        self.pairwise = pairwise
        self.psqt = psqt
        self.embed = nn.EmbeddingBag(NUM_FEATURES * king_buckets, hidden, mode="sum")
        self.bias1 = nn.Parameter(torch.zeros(hidden))
        self.out = nn.Linear(hidden if pairwise else 2 * hidden, buckets)
        nn.init.normal_(self.embed.weight, std=0.05)
        nn.init.normal_(self.out.weight, std=0.1)
        nn.init.zeros_(self.out.bias)
        # The skip head starts at exactly zero, so epoch 0 is the no-PSQT net and material is
        # learnt into it rather than initialised as noise the positional half has to cancel.
        self.psqt_embed = nn.EmbeddingBag(NUM_FEATURES * king_buckets, buckets, mode="sum")
        nn.init.zeros_(self.psqt_embed.weight)

    def arch(self) -> str:
        """The string the engine's ``load_weights`` checks: it cannot tell a pairwise dual net
        from a single-perspective one by shape alone, so the shape check is not enough."""
        core = "pairwise" if self.pairwise else ("screlu" if self.screlu else "crelu")
        return f"dual-shared-{core}" + ("-psqt" if self.psqt else "")

    def merged_input_weight(self) -> torch.Tensor:
        """The input table the engine sees (no factoriser on this architecture)."""
        return self.embed.weight

    def clip_(self, clip: float) -> None:
        """Keep every input weight within +/-clip (the int16 bound is checked at export)."""
        self.bias1.clamp_(-clip, clip)
        self.embed.weight.clamp_(-clip, clip)

    def _perspective(self, flat: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        hidden = torch.clamp(self.embed(flat, offsets) + self.bias1, 0.0, 1.0)
        if self.pairwise:
            half = self.hidden // 2
            return hidden[:, :half] * hidden[:, half:]
        return hidden * hidden if self.screlu else hidden

    def forward(
        self,
        flat0: torch.Tensor,
        flat1: torch.Tensor,
        offsets: torch.Tensor,
        bucket: torch.Tensor,
    ) -> torch.Tensor:
        both = torch.cat([self._perspective(flat0, offsets), self._perspective(flat1, offsets)], 1)
        out = self.out(both)
        if self.psqt:
            # own perspective minus opponent's, for every bucket at once; the gather below picks
            # this position's bucket out of it exactly as it does for the positional output
            out = out + self.psqt_embed(flat0, offsets) - self.psqt_embed(flat1, offsets)
        if self.buckets == 1:
            return out.squeeze(1)
        return out.gather(1, bucket.unsqueeze(1)).squeeze(1)


def bucket_of(counts: np.ndarray, buckets: int) -> np.ndarray:
    """Output-bucket index from the piece count, exactly as the engine computes it."""
    if buckets == 1:
        return np.zeros(len(counts), dtype=np.int64)
    return np.clip((counts - 2) // BUCKET_DIVISOR, 0, buckets - 1).astype(np.int64)


def decode_dual(
    occ: np.ndarray, nib: np.ndarray, rows: np.ndarray, king_buckets: int, buckets: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flat0, flat1, offsets = decode_batch_dual(occ, nib, rows, king_buckets)
    counts = np.diff(np.append(offsets, len(flat0)))
    return flat0, flat1, offsets, bucket_of(counts, buckets)


def quantise_dual(model: NnueDual) -> dict[str, np.ndarray]:
    weight1 = model.merged_input_weight().detach().cpu().numpy().astype(np.float64)
    bias1 = model.bias1.detach().cpu().numpy().astype(np.float64)
    weight2 = model.out.weight.detach().cpu().numpy().astype(np.float64)  # (buckets, in_width)
    bias2 = model.out.bias.detach().cpu().numpy().astype(np.float64)  # (buckets,)
    weights = {
        "w1": np.rint(weight1 * QA).astype(np.int16),
        "b1": np.rint(bias1 * QA).astype(np.int16),
        "w2": np.rint(weight2 * QB).astype(np.int16),
        "b2": np.rint(bias2 * QA * QB).astype(np.int32),
        "arch": np.array(model.arch()),
    }
    if model.psqt:
        psqt = model.psqt_embed.weight.detach().cpu().numpy().astype(np.float64)
        scaled = np.rint(psqt * QA * QB)
        # b2's scale, because it is b2's kind of term. int32 per weight is the only bound that
        # matters -- the engine's running PSQT sum is int64, so 32 rows cannot overflow it.
        if np.abs(scaled).max() > np.iinfo(np.int32).max:
            raise SystemExit(
                f"PSQT weight {np.abs(psqt).max():.3f} does not fit int32 at scale QA*QB"
            )
        weights["wp"] = scaled.astype(np.int32)
    return weights


def quantised_forward_dual(
    weights: dict[str, np.ndarray],
    flat0: np.ndarray,
    flat1: np.ndarray,
    offsets: np.ndarray,
    bucket: np.ndarray,
    count: int,
    screlu: bool = True,
) -> np.ndarray:
    """Integer forward pass in numpy, bit-identical to ``nnue_eval.nnue_evaluate_into``.

    The architecture is read off the weights rather than passed in, so this cannot disagree with
    the file it is handed: ``w2`` as wide as the accumulator means pairwise, and a ``wp`` key means
    the PSQT head. Only SCReLU-vs-CReLU has to be told, since those share a shape."""
    w1, b1, w2, b2 = weights["w1"], weights["b1"], weights["w2"], weights["b2"]
    hidden = b1.shape[0]
    pairwise = w2.shape[1] == hidden
    half = hidden // 2
    wp = weights.get("wp")
    out = np.empty(count, dtype=np.int64)
    ends = np.append(offsets[1:], len(flat0))
    for i in range(count):
        rows0 = flat0[offsets[i] : ends[i]]
        rows1 = flat1[offsets[i] : ends[i]]
        acc0 = b1.astype(np.int32) + w1[rows0].sum(axis=0, dtype=np.int32)
        acc1 = b1.astype(np.int32) + w1[rows1].sum(axis=0, dtype=np.int32)
        acc0 = np.clip(acc0, 0, QA).astype(np.int64)
        acc1 = np.clip(acc1, 0, QA).astype(np.int64)
        if pairwise:
            acc0 = acc0[:half] * acc0[half:] // QA
            acc1 = acc1[:half] * acc1[half:] // QA
        elif screlu:
            acc0 = acc0 * acc0 // QA
            acc1 = acc1 * acc1 // QA
        row = w2[bucket[i]].astype(np.int64)
        width = row.shape[0] // 2
        total = int(b2[bucket[i]])
        total += int((acc0 * row[:width]).sum()) + int((acc1 * row[width:]).sum())
        if wp is not None:
            # own perspective minus opponent's: zero in a colour-symmetric position
            total += int(wp[rows0, bucket[i]].astype(np.int64).sum())
            total -= int(wp[rows1, bucket[i]].astype(np.int64).sum())
        out[i] = total
    return out * CP_SCALE // (QA * QB)


def report_dual(
    model: NnueDual,
    weights: dict[str, np.ndarray],
    occ: np.ndarray,
    nib: np.ndarray,
    scores: np.ndarray,
    rows: np.ndarray,
    quant_sample: int,
    king_buckets: int,
) -> dict[str, float]:
    """Held-out centipawn MAE / correlation for the float net and for the quantised net.
    Identical metric definitions to ``train.report`` so the two are directly comparable."""
    model.eval()
    device = next(model.parameters()).device
    predictions = np.empty(len(rows), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(rows), 65536):
            chunk = rows[start : start + 65536]
            flat0, flat1, offsets, bucket = decode_dual(
                occ, nib, chunk, king_buckets, model.buckets
            )
            value = model(
                torch.from_numpy(flat0).to(device),
                torch.from_numpy(flat1).to(device),
                torch.from_numpy(offsets).to(device),
                torch.from_numpy(bucket).to(device),
            )
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
    flat0, flat1, offsets, bucket = decode_dual(occ, nib, sample, king_buckets, model.buckets)
    quant_cp = quantised_forward_dual(
        weights, flat0, flat1, offsets, bucket, len(sample), model.screlu
    ).astype(np.float32)
    quant_cp = np.clip(quant_cp, -2000, 2000)
    metrics["quant_mae_cp"] = float(np.abs(quant_cp - scores[sample]).mean())
    metrics["quant_corr"] = float(np.corrcoef(quant_cp, scores[sample].astype(np.float32))[0, 1])
    metrics["quant_vs_float_mae_cp"] = float(np.abs(quant_cp - float_cp[:quant_sample]).mean())
    metrics["quant_vs_float_max_cp"] = float(np.abs(quant_cp - float_cp[:quant_sample]).max())
    quiet = np.abs(target) < 400
    metrics["float_mae_cp_under400"] = float(np.abs(float_cp[quiet] - target[quiet]).mean())
    metrics["float_sign_agreement"] = float(np.mean((float_cp > 0) == (target > 0)))
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("training/data/positions_bal.npy"))
    parser.add_argument("--hidden", type=int, default=512, help="units per perspective (N)")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--holdout", type=int, default=500_000)
    parser.add_argument("--train-limit", type=int, default=0, help="0 = all remaining positions")
    parser.add_argument("--quant-sample", type=int, default=100_000)
    parser.add_argument("--clip", type=float, default=1.0, help="first-layer weight clip")
    parser.add_argument("--out", type=Path, default=Path("training/data/nnue_dual_h512.npz"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("training/data/nnue_dual_h512.pt")
    )
    parser.add_argument(
        "--curve", type=Path, default=Path("training/data/curve_dual_h512.json")
    )
    parser.add_argument("--quantise-only", action="store_true")
    parser.add_argument("--buckets", type=int, default=8,
                        help="output buckets chosen by piece count ((pieces - 2) // 4)")
    parser.add_argument("--no-screlu", dest="screlu", action="store_false",
                        help="plain clipped ReLU instead of SCReLU (SCReLU is the default here)")
    parser.add_argument("--pairwise", action="store_true",
                        help="Stockfish's pairwise multiplication instead of SCReLU: clip each "
                             "perspective to [0, 1] and multiply its two halves, so the readout "
                             "sees N terms per position pair instead of 2N and gains cross-terms")
    parser.add_argument("--psqt", action="store_true",
                        help="add a per-bucket PSQT skip head read straight off the features, "
                             "own perspective minus opponent's")
    parser.add_argument("--loss-exp", type=float, default=2.0,
                        help="loss = |sigmoid(v) - t|^exp")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--schedule", choices=("step", "cosine"), default="step",
                        help="step: x0.7 per epoch; cosine: lr -> lr/100")
    parser.add_argument(
        "--king-buckets", type=int, default=8,
        help="1 = plain 768 inputs; 8 = mirror each perspective to its own king's half and offset "
             "the features by that king's bucket (encoding.KING_BUCKET); 32 = per-square",
    )
    parser.add_argument(
        "--wdl", type=float, default=0.0,
        help="weight of the game result in the target: (1-wdl)*sigmoid(score/CP_SCALE) + "
             "wdl*(result+1)/2; needs a set built with build_set.py --with-result",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device for the float training (the data pipeline stays on the CPU)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

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
    # The same seed, permutation and holdout size as train.py, so a dual net's held-out numbers
    # are directly comparable to the single-perspective net's on the same file.
    rng = np.random.default_rng(20260904)
    order = rng.permutation(total)
    holdout = order[: args.holdout]
    train_rows = order[args.holdout :]
    if args.train_limit:
        train_rows = train_rows[: args.train_limit]
    print(f"{total:,} positions: {len(train_rows):,} train / {len(holdout):,} held out")

    device = torch.device(args.device)
    print(f"device {device}", flush=True)
    model = NnueDual(
        args.hidden, args.king_buckets, args.buckets, args.screlu, args.pairwise, args.psqt
    )
    king_buckets = args.king_buckets
    params = sum(p.numel() for p in model.parameters())
    print(
        f"dual-perspective: w1 {NUM_FEATURES * king_buckets} x {args.hidden}, "
        f"w2 {args.buckets} x {model.out.in_features}, {params:,} parameters, "
        f"arch={model.arch()}",
        flush=True,
    )
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
                flat0, flat1, offsets, bucket = decode_dual(
                    occ, nib, chunk, king_buckets, args.buckets
                )
                value = model(
                    torch.from_numpy(flat0).to(device),
                    torch.from_numpy(flat1).to(device),
                    torch.from_numpy(offsets).to(device),
                    torch.from_numpy(bucket).to(device),
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
            weights = quantise_dual(model)
            metrics = report_dual(
                model, weights, occ, nib, scores, holdout[:200_000], 20_000, king_buckets
            )
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

    weights = quantise_dual(model)
    w1_wide, b1_wide = weights["w1"].astype(np.int64), weights["b1"].astype(np.int64)
    bound = accumulator_bound(w1_wide, b1_wide)
    print(f"worst-case |accumulator| = {bound} over legal placements (int16 limit 32767; "
          f"any-32-pieces bound {accumulator_bound_any32(w1_wide, b1_wide)})")
    if bound > 32767:
        raise SystemExit("accumulator would overflow int16: lower --clip and retrain")
    if "wp" in weights:
        psqt_bound = int(np.abs(weights["wp"].astype(np.int64)).max()) * MAX_PIECES
        print(f"worst-case |psqt accumulator| = {psqt_bound} "
              f"(int64 in the engine, so the only real limit is int32 per weight: "
              f"{int(np.abs(weights['wp']).max())} of {np.iinfo(np.int32).max})")
    np.savez_compressed(args.out, **weights)
    metrics = report_dual(
        model, weights, occ, nib, scores, holdout, args.quant_sample, king_buckets
    )
    print(json.dumps(metrics, indent=2))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
