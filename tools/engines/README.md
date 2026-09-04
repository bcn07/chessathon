# Reference engines (local only, gitignored)

Binaries the `opponents/*` directories point at. None of this ships. `make engines` builds them:

| binary | source | why |
|---|---|---|
| `stockfish` | Homebrew | `UCI_LimitStrength` at 1400/1800/2200/2600 gives an absolute Elo yardstick |
| `shallow-blue` | github.com/GunshipPenguin/shallow-blue | house bot, CCRL 1576 |
| `rustic` | github.com/mvanthoor/rustic (Alpha 3) | house bot, CCRL 1792 |
| `zagreus` | github.com/Dannyj1/Zagreus (5.0) | house bot, CCRL 2168 |
| `loki` | github.com/BimmerBass/Loki (3.0) | house bot, CCRL 2428 |

Status on Apple Silicon: Stockfish, Shallow Blue and Loki build and run. Zagreus does not — it
uses x86 intrinsics (`ia32intrin.h`) and `-static-libgcc`; use its Linux release on the cluster.
Rustic needs `cargo` (`brew install rust`, then `make engines` again).
