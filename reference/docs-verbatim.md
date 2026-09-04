# Verbatim extract — https://aichessathon.com/docs

Fetched 2026-09-04. Text extracted from the page HTML (tags stripped); wording unedited.
See ../RULES.md for the digest.

---
 Docs | AI Chessathon 
 Skip to content 
 AI Chessathon Docs Leaderboard Daily Five Rules Docs Leaderboard Daily Five Rules Sign in Sign in Register 
 Participants
 Build against this.
 Quickstart Agent API Wire protocol Match environment Clocks and scoring Submissions Failure reference Quickstart
 Fork the starter repo for a working agent, baselines to beat, and a harness that plays games locally under the real clock. Or start from scratch, below.
 A submission is a zip of 50 MB unzipped at most. These sit at its root, not inside a folder.
 agent.zip
├── agent.py required
└── model files weights, books, anything else The smallest legal agent looks like this.
 import chess
import random
def get_move(fen: str, time_left_ms: int) -> str:
 board = chess.Board(fen)
 return random.choice(list(board.legal_moves)).uci() The environment is fixed. The container ships Python 3.12, the full standard library, and five preinstalled packages at fixed versions.
 torch 2.13.0+cpu numpy 2.5.2 python-chess 1.11.2
onnxruntime 1.29.0 numba 0.67.0 Nothing installs at validation and a requirements.txt in your zip is ignored, so importing anything outside that set crashes your agent in its smoke games. Ask at hello@aichessathon.com for a package the stack lacks and any addition is announced to every team.
 Native binaries inside the zip are rejected. What you ship has to be source a judge can read, so a flagged game can be cleared by reading your agent instead of by statistics alone. Compiled speed comes from numba, which JIT compiles your Python in process, and each jitted function pays that compile cost the first time it runs. Cython does not work here, because a compiled extension is a native binary and native binaries are rejected, and the image carries no compiler to build one at runtime. Your zip goes first on sys.path , so a file named after a module you import, chess.py or types.py , shadows the real one.
 Agent API
 agent.py exposes one function.
 def get_move(fen: str, time_left_ms: int) -> str Name | Type | Meaning | 
 fen | str | the position to move in, standard FEN | 
 time_left_ms | int | your remaining clock in milliseconds | 
 returns | str | a legal move in UCI, e.g. e2e4 or e7e8q | 
 One process serves one game, started fresh for every game. Load your model at import. The 60s init budget covers importing your agent and runs before the clock starts, so work you defer to your first get_move comes out of your match clock. The process stays alive between moves, so state you keep in memory carries across your own moves. The process keeps its dedicated core after get_move returns, and pondering is allowed. During your own move one thread is fastest, since threads past the first share the single core and cost you time. The referee claims threefold and fifty-move draws automatically, so an agent that wants to avoid a repetition tracks the positions it has been asked about.
 Wire protocol
 The runner talks to your process over stdin and stdout, one JSON object per line. You only implement get_move . The provided runner handles the wire. Each request looks like this.
 {"fen": "rnbqkbnr/...", "time_left_ms": 87500} Each response looks like this.
 {"move": "e2e4"} your colour | the side to move in the fen | 
 time_left_ms | your clock before this move. The increment lands after you move | 
 output cap | 4 KB per move. Past it the game is lost | 
 malformed output | counts as an illegal move, which loses the game | 
 Your own output cannot corrupt this. The runner moves the protocol onto a private handle and points file descriptor 1 at stderr before importing your agent, so print is safe. Everything you write to stdout or stderr is discarded during rated games and shown back to you in the validation log, up to 8 KB.
 Match environment
 CPU | 1 dedicated core | 
 Memory | 2 GB | 
 Network | none, in either direction | 
 GPU | none | 
 Filesystem | read-only, plus 256 MB scratch at /tmp , where HOME and the cache paths already point. It starts empty for every game and is deleted with it, so it is scratch space, not a cache between games | 
 Processes | 128. On one core, threads past the first cost you time | 
 Hardware | identical for every game, both agents on one machine | 
 No network means no hosted inference and no engine APIs, by construction. Everything your agent needs ships inside the zip.
 Clocks and scoring
 Time control | 120s per side, plus 0.5s per move | 
 Init budget | 60s before the clock starts | 
 Game end | FIDE rules. 300 plies without a result is adjudicated on material, else drawn | 
 Ladder | rated rounds every hour, 08:00 to 22:00, ranked by Elo. The ladder only seeds the final Swiss | 
 Openings | every game starts from a curated position that is close to level. Knockout ties play each position once with each colour | 
 Qualification | a 13-round Swiss over the locked builds decides the order the 50 London seats fill, by points. An odd field gives one team a 1-point bye. Seats go one per UK member and at most two per team | 
 Prizes | £1,000 winner, £500 runner-up, £250 third at the London final. Third is the losing semi-finalist with the better final Swiss standing. Awarded to the team, split as the team decides | 
 Tie-breaks | points, Buchholz, head-to-head, earlier final submission | 
 Daily Five | 6 to 10 September. One attempt a day, five positions, 20 minutes. One wrong move ends a position. Your five are drawn for you. Start by 23:40 London. Anyone signed in may play | 
 Daily Five ranking | positions solved, then the time of your last solve, then who finished first. Giving up costs no time. A position withdrawn as broken counts as solved and adds no time | 
 Daily Five wildcards | the top 3 eligible participants each day earn a Finals Day Wildcard, a seat at the London final and not a place in the bracket. One per person across the five days, so places roll down. Eligible means a UK university student who is not an organiser and not on a disqualified team | 
 Daily Five fair play | no engines, no other people, no other accounts, no looking positions up. Every move is timed. Invitations follow review and may include solving a position in person at the final | 
 Teams | 1 to 3 people, one team per person. Creating, joining and leaving a team close 11 September 11:00 | 
 Eligibility | the ladder is open worldwide. A team enters the final Swiss if at least one member is a UK university student, and only its UK members can take a London seat | 
 House bots | play the ladder and show their public CCRL ratings. They cannot qualify | 
 Submissions
 Size | 50 MB unzipped | 
 Rate | 6 uploads per team per day | 
 Which plays | your latest submission that passed validation | 
 Validation | build, then two smoke games, one as each colour. The verbatim log appears on your dashboard | 
 Uploads close | 11 September 11:00 | 
 Third party engines are prohibited. That means Stockfish, Lc0, Maia and any wrapper around one. Your moves come from code you wrote and any model you ship is one you trained. An engine you wrote yourself before the event is your own code. A model is not required, a classical search is a full entry. Training data is unrestricted, including positions annotated by an existing engine. The ban covers only what ships inside the submission.
 What you ship has to be source a judge can read, so a flagged game can be cleared by reading your agent instead of by statistics alone. Model weights are not binaries, so .onnx , .safetensors and .pt are fine. Books and tablebases are permitted as shipped data, and chess.polyglot and chess.syzygy are in the base image. Every submission faces automated and human checks. Each finalist team walks through how its agent was built. Disqualification can be retroactive, and obfuscated agents are disqualified.
 Failure reference
 Termination | Cause | Result | 
 illegal | illegal or malformed move, or output past the cap | loss | 
 crash | your process exited, threw, or ran out of memory | loss | 
 flag | clock ran out mid-game | loss | 
 init | no ready line within the 60s init budget | loss | 
 adjudication | 300 plies without a result | material decides, else draw | 
 void | both sides failed | no result recorded | 
 AI Chessathon Sponsored by 
 Event Format Team Register Archive 
 Legal Competition rules Privacy notice 
 Contact Email us 
 © 2026 AI Chessathon Made by Advit Arora and Arham Shuaib 
 