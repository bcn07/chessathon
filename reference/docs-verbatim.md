# Verbatim extract — https://aichessathon.com/docs

Fetched 2026-09-12 11:27 (finals day; supersedes the 2026-09-04 copy). Text extracted from the page HTML (tags stripped); wording unedited.
See ../RULES.md for the digest.

---

Docs | AI Chessathon
Skip to content
AI Chessathon
Docs
Leaderboard
Daily Five
Rules
Docs
Leaderboard
Daily Five
Rules
Sign in
Sign in
Register
Participants
Build against this.
Quickstart
What you submit
Your process
Match conditions
Uploads
Dates
Teams and eligibility
Ranking and seats
Daily Five
What you may ship
Quickstart
Fork
the starter repo
for a working agent, baselines to beat, and a harness that plays games locally under the real clock. Or start from scratch, below.
A submission is a zip of at most 50 MB unzipped. These sit at its root, not inside a folder.
agent.zip
├── agent.py            required
└── model files         weights, books, anything else
The smallest legal agent looks like this.
import chess
import random
def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    return random.choice(list(board.legal_moves)).uci()
The environment is fixed. Python 3.12, the full standard library, and five preinstalled packages at these versions.
torch 2.13.0+cpu     numpy 2.5.2      python-chess 1.11.2
onnxruntime 1.29.0   numba 0.67.0
What you submit
The zip
agent.py
 at the root, exposing
get_move(fen: str, time_left_ms: int) -> str
 and returning a UCI move such as 
e2e4
 or 
e7e8q
.
Model weights, opening books and any other files, at most 50 MB unzipped in total.
The root means the top of the archive, not a folder inside it. Most zip tools wrap the folder you selected, so check before uploading. A zip without
agent.py
 at its root is rejected.
Name
Type
Meaning
fen
str
the position to move in, standard FEN
time_left_ms
int
your remaining clock in milliseconds
returns
str
a legal move in UCI, e.g. 
e2e4
 or 
e7e8q
The environment
Nothing else installs and a 
requirements.txt
 in the zip is ignored. An import outside that set crashes your agent in its smoke games.
Ask 
hello@aichessathon.com
 for a package the stack lacks. Any addition is announced to every team.
Compiled speed comes from the stack, not from your zip. numba JIT compiles your Python in process, and each jitted function pays that compile cost the first time it runs.
Cython does not work here. A compiled extension is a native binary, which is rejected, and the image carries no compiler to build one at runtime.
What may be in the zip
Native binaries in the zip are rejected. Model weights are not binaries, so
.onnx
, 
.safetensors
 and 
.pt
 are fine.
What you ship has to be source a judge can read, so a flagged game can be cleared by reading your agent instead of by statistics alone.
Any network you ship is one you trained yourself. Starting from a published chess network is not allowed, so fine-tuning or re-exporting one counts as shipping it.
File modes inside the zip are ignored and every file is readable by your process.
Your zip is first on 
sys.path
, so a file named after a module you import, like 
chess.py
 or 
types.py
, shadows the real one.
Your process
How it runs
One process per agent per game, started fresh for every game.
Load your model at import. The init budget covers importing your agent and runs before the clock starts. It is 90 s in the qualifier and 30 s at the final. Work you defer to your first
get_move
 comes out of your match clock instead.
The process stays alive between moves, so state you keep in memory carries across your own moves. It does not carry to your next game.
Your process is suspended while your opponent thinks, so work you leave running between your own moves does not run. Each side has the core to itself while it thinks.
During your own move one thread is fastest. Threads past the first share the single core and cost you time.
Games are independent, so two of your games can run at the same time in separate containers. Your agent is never asked for two moves at once.
The referee claims threefold and fifty-move draws automatically, so an agent that wants to avoid a repetition tracks the positions it has been asked about.
Wire protocol
The runner talks to your process over stdin and stdout, one JSON object per line. You only implement 
get_move
. The runner handles the wire. Each request looks like this.
{"fen": "rnbqkbnr/...", "time_left_ms": 87500}
Each response looks like this.
{"move": "e2e4"}
time_left_ms
 is your remaining clock before this move. The increment lands after you move.
Your colour is the side to move in the fen.
The first fen you receive is the starting position of the game. Repetition and fifty-move counts begin there.
Every game starts from a curated opening position that is close to level. The set is not published. Finished games reveal the positions they were played from.
Your output and your log
Your own output cannot corrupt the protocol. The runner moves the protocol onto a private handle and points file descriptor 1 at stderr before importing your agent, so 
print
 is safe.
Everything you write to stdout or stderr is kept, up to 8 KB as the first 4 KB and the last 4 KB.
It appears in your validation log, and after every rated game in a log your dashboard offers alongside the PGN.
That log also carries your init time, your time on every move, and the clock you had left.
Only your own team can read it.
Match conditions
Time control
120 s plus 0.5 s per move, per side, on wall time
Init budget
90 s in the qualifier, 30 s at the final, to import your bot
Hardware
one core of an AMD EPYC 9V74, measured at 2.60 GHz. 2 GB RAM. No network. No GPU. Identical for every game
Machine
both agents of a game run on the same machine and take the core in turns
Filesystem
read-only apart from 256 MB at 
/tmp
. 
HOME
, 
TORCH_HOME
, 
HF_HOME
 and the other cache paths already point there
Scratch
/tmp
 starts empty for every game and is deleted with the game, so use it as scratch space, not as a cache between games
Processes
at most 128 processes and threads alive at once. On one core, threads past the first cost you time
No network means no hosted inference and no engine APIs, by construction. Everything your agent needs ships inside the zip.
How a game ends
An illegal move, malformed output, a crash, running out of memory or missing the init budget loses the game. A move payload over 4 KB counts as an illegal move.
Losing on time loses unless the other side has no way to mate, and then the game is a draw.
If both sides fail the game is void. There are no retries within a game.
Draws follow FIDE rules. A game still running at 600 plies is a draw, and the opening position counts toward the 600.
The FIDE rules run through python-chess, which covers stalemate, threefold repetition, the fifty move rule and insufficient material.
Every game starts from a curated opening position that is close to level. The set is not published.
Knockout ties play each position once with each colour. Two positions a tie, three in the semis and four in the final.
Termination codes
illegal
an illegal or malformed move, or a move payload over 4 KB
crash
your process exited, threw, or ran out of memory
flag
the clock ran out
init
no ready line within the init budget
ply_cap
the game reached 600 plies
void
both sides failed, so no result is recorded
Uploads
Size
at most 50 MB unzipped
Uploads
30 per team per day
Which plays
the latest upload that passed validation
Validation
build, then two smoke games at the match clock against a house agent, one as each colour, from curated opening positions. The verbatim log appears on your dashboard
Uploads close
11 September 11:00
Uploads reopen
12 September 10:30 to 14:00, for the London final
Your last valid build then freezes. For eligible teams the frozen build alone plays the final qualification Swiss, which does not open until every submission has finished validating.
Dates
Registration
open now, closes 11 September 11:00
Qualifier ladder
4 to 11 September
Rated rounds
every hour, 08:00 to 22:00, from 4 September 08:00
Daily Five
6 to 10 September
Uploads close
11 September 11:00
Uploads reopen
12 September 10:30 to 14:00, for the London final
Team changes close
11 September 11:00. Creating, joining and leaving a team all stop
Final qualification
11 September afternoon, a 13-round Swiss over locked builds
Finalist invites
11 September, once the final Swiss has run
Live final
12 September at Encode Club, London
Teams and eligibility
Teams are 1 to 3 people, and a person is on one team.
Entry is open worldwide.
A team enters the final Swiss if at least one member is a UK university student.
Only a team's UK university students can take a London seat, and that is verified before invites go out.
Ranking and seats
Ranking and qualification
The ladder ranks by a standard rating fitted to your current build's results.
The ladder only seeds the final Swiss.
House bots and house engines play the ladder. The engines keep a fixed rating, which holds the scale. None can qualify.
Only the locked-build final Swiss counts for qualification, by points. An odd field gives one team a 1-point bye.
Tie-breaks are points, then Buchholz, then head-to-head, then the earlier final submission.
A level knockout tie goes to the better final Swiss standing.
London seats
The final Swiss decides the London field.
The room holds 50 people and a person takes one seat.
Seats fill in seed order, one per UK university student on a team.
Invites go out in that order and are confirmed by reply, first come, until the room is full.
London final
Uploads reopen 10:30 to 14:00 on the day for confirmed seats, and the build you have at 14:00 plays the knockout.
The init budget is 30 s at the final and every upload in the window validates against it.
Practice rounds run every ten minutes from 10:40 to 13:50 on the latest valid builds. They count for nothing.
The bracket seeds from the final Swiss standings and the top seeds take a first-round bye.
Prizes
£1,000 to the winner, £500 to the runner-up and £250 to third, paid at the London final. Third is the losing semi-finalist with the better final Swiss standing. A prize goes to the team, split as the team decides.
Daily Five
6 to 10 September. One attempt a day, five positions, 20 minutes.
One wrong move ends a position.
Your five are drawn for you.
Start by 23:40 London.
Anyone signed in may play.
Ranking
Positions solved, then the time of your last solve, then who finished first.
A position withdrawn as broken counts as solved and adds no time.
Wildcards
The top 3 eligible participants each day earn a Finals Day Wildcard. That is a seat at the London final, not a place in the bracket.
One per person across the five days, so places roll down.
A wildcard won by someone who also takes a seat through the final Swiss rolls down the same way.
Eligible means a UK university student who is not an organiser and not on a disqualified team.
Fair play
No engines, no other people, no other accounts, no looking positions up.
Every move is timed.
Invitations follow review and may include solving a position in person at the final.
What you may ship
Engines
Third party engines are prohibited. That covers Stockfish, Lc0, Maia, any wrapper around one and any port or translation of one.
Your moves come from code you wrote. An engine you wrote yourself before the event is your own code.
Use any AI support you like to write it, as long as the submission keeps to these rules and you can explain it when asked.
A model is not required. A classical search is a full entry.
Models and training data
Any network you ship is one you trained yourself. Starting from a published chess network is not allowed, so fine-tuning or re-exporting one counts as shipping it.
Training data is unrestricted, including positions annotated by an existing engine. What ships inside the zip is what the ban covers.
Books and tablebases
A table you ship and read during a game may answer the opening or the endgame. The middlegame you search yourself.
The opening is a position whose move number is 20 or lower. The endgame is a position of at most 7 pieces, counting both kings.
Both are read from the position you are given, and neither depends on what produced the table.
A table that answers a middlegame position is a stored search and counts as an engine.
Books and tablebases count against the 50 MB cap.
chess.polyglot
 and 
chess.syzygy
 are in the base image.
Code
What you ship must be source a judge can read. Obfuscated agents are disqualified.
Everything that runs is Python from your zip plus the preinstalled stack.
Verification
Every submission faces automated and human checks.
Each finalist team walks through how its agent was built, and a team that ships a network shows how it was trained.
Disqualification can be retroactive.
AI Chessathon
Sponsored by
Event
Format
Team
Register
Archive
Legal
Competition rules
Privacy notice
Contact
Email us
Discord
LinkedIn
