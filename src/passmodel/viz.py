"""mplsoccer plots: passes shown as football (PLAN.md step 6).

Colours are fixed project-wide: attacking (passing) team blue, defending team red, ball black.
Pitch: mplsoccer "skillcorner" type = metres with origin at the centre spot, matching clean.py.
"""

import matplotlib.pyplot as plt
import numpy as np
from mplsoccer import Pitch

ATTACK_COLOUR, DEFEND_COLOUR, BALL_COLOUR = "#1f5fbf", "#d62728", "black"


def plot_pass_frame(pass_row, frame, ax=None, title=None):
    """Draw one pass at its start frame: all players, velocity arrows (1 s ahead), ball, event start->end.

    pass_row: one row of passes.parquet (passing team attacks +x, left to right).
    frame:    tracking rows for pass_row.start_frame (metres, home attacks +x); rotated here by attack_sign.
    The arrow from the event start to the event end is drawn for sanity checks only; it is
    outcome-dependent and never a model input.
    """
    pitch = Pitch(pitch_type="skillcorner", pitch_length=105, pitch_width=68, line_color="#888888")
    if ax is None:
        _, ax = pitch.draw(figsize=(9, 6))
    else:
        pitch.draw(ax=ax)
    s = pass_row["attack_sign"]
    passing_team = "home" if s == 1 else "away"
    for team, colour in [(passing_team, ATTACK_COLOUR), ({"home": "away", "away": "home"}[passing_team], DEFEND_COLOUR)]:
        t = frame[frame["team"] == team]
        x, y, vx, vy = (t[c].to_numpy() * s for c in ("x", "y", "vx", "vy"))
        pitch.scatter(x, y, ax=ax, s=90, color=colour, edgecolors="white", zorder=3)
        pitch.arrows(x, y, x + np.nan_to_num(vx), y + np.nan_to_num(vy), ax=ax, color=colour, width=1.2,
                     headwidth=4, alpha=0.7, zorder=2)
    passer = frame[frame["player_id"] == pass_row["passer"]]
    pitch.scatter(passer["x"] * s, passer["y"] * s, ax=ax, s=260, facecolors="none", edgecolors=ATTACK_COLOUR,
                  linewidths=2, zorder=4)
    ball = frame[frame["player_id"] == "ball"]
    pitch.scatter(ball["x"] * s, ball["y"] * s, ax=ax, s=40, color=BALL_COLOUR, zorder=5)
    pitch.arrows(pass_row["start_x"], pass_row["start_y"], pass_row["end_x"], pass_row["end_y"], ax=ax,
                 color="#444444", width=1.5, linestyle="--", zorder=1)
    outcome = "completed" if pass_row["completed"] else "failed"
    ax.set_title(title or f"Game {pass_row['match_id']}, frame {pass_row['start_frame']}: {pass_row['passer']} "
                          f"pass, {outcome}. Blue attack left to right", fontsize=10)
    return ax


def plot_sanity_passes(passes, read_frames, n=3, seed=0, path=None):
    """Plot n random passes side by side to eyeball that event frames and tracking line up."""
    sample = passes.sample(n, random_state=seed)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5.5))
    for ax, (_, row) in zip(np.atleast_1d(axes), sample.iterrows()):
        frame = read_frames(row["match_id"], [row["start_frame"]])
        plot_pass_frame(row, frame, ax=ax)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return fig
