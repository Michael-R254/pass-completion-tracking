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

    pass_row: one row of passes.parquet (passing team attacks +x, left to right). If it also has
              target_x / target_y (from features.parquet), the intended target (D3) is ringed and the
              passer->target line the model scored is drawn solid.
    frame:    tracking rows for pass_row.start_frame (metres, home attacks +x); rotated here by attack_sign.
    The dashed arrow from the event start to the event end is where the ball actually went; it is
    outcome-dependent, shown for context only, and never a model input.
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
    if "target_x" in pass_row:
        start, target = (pass_row["start_x"], pass_row["start_y"]), (pass_row["target_x"], pass_row["target_y"])
        pitch.lines(*start, *target, ax=ax, color=ATTACK_COLOUR, lw=3, alpha=0.6, zorder=1)
        pitch.scatter(*target, ax=ax, s=420, facecolors="none", edgecolors=ATTACK_COLOUR, linewidths=3, zorder=4)
        pitch.annotate("most likely target", (target[0], target[1] + 3), ax=ax, ha="center", fontsize=8,
                       color=ATTACK_COLOUR, zorder=6)
    outcome = "completed" if pass_row["completed"] else "failed"
    ax.set_title(title or f"Game {pass_row['match_id']}, frame {pass_row['start_frame']}: {pass_row['passer']} "
                          f"pass, {outcome}. Blue attack left to right", fontsize=10)
    return ax


def plot_passes(rows, read_frames, titles=None, path=None):
    """Plot the passes in `rows` side by side, one pitch each, reading only their start frames."""
    titles = titles or [None] * len(rows)
    fig, axes = plt.subplots(1, len(rows), figsize=(7 * len(rows), 5.5))
    for ax, (_, row), title in zip(np.atleast_1d(axes), rows.iterrows(), titles):
        plot_pass_frame(row, read_frames(row["match_id"], [row["start_frame"]]), ax=ax, title=title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return fig


def plot_sanity_passes(passes, read_frames, n=3, seed=0, path=None):
    """Plot n random passes side by side to eyeball that event frames and tracking line up."""
    return plot_passes(passes.sample(n, random_state=seed), read_frames, path=path)


def pass_story_title(row) -> str:
    """Coach-readable title: who passed, when, what the model said, what happened."""
    outcome = "came off" if row["completed"] else "was lost"
    return (f"Game {row['match_id']}, ~{row['approx_minute']:.0f}': {row['passer']} to {row['target_player']} "
            f"(most likely target)\nModel: {row['p_complete']:.0%} chance of completion. It {outcome}. "
            f"Blue attacks left to right; arrows = 1 s of running")


def plot_calibration(table, path=None, title=None):
    """Reliability curve from evaluate.calibration_table: predicted vs observed completion, with bin counts.

    Points on the dashed diagonal = honest probabilities. Each point is one equal-count bin of held-out passes.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1, label="perfectly calibrated")
    ax.plot(table["mean_predicted"], table["observed_rate"], marker="o", markersize=8, linewidth=2,
            color=ATTACK_COLOUR, label="model (held-out matches)")
    for _, r in table.iterrows():
        ax.annotate(f"n={int(r['n'])}", (r["mean_predicted"], r["observed_rate"]), textcoords="offset points",
                    xytext=(6, -12), fontsize=8, color="#555555")
    lo = min(table["mean_predicted"].min(), table["observed_rate"].min()) - 0.05
    ax.set(xlim=(lo, 1), ylim=(lo, 1), xlabel="predicted chance of completion",
           ylabel="share of passes actually completed")
    ax.set_title(title or "When the model says X%, do X% of passes come off?", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return fig


def plot_shap_summary(values, X, path=None, max_display=12):
    """SHAP beeswarm: one dot per held-out pass per feature; right = raises completion chance (log-odds)."""
    import shap  # imported here: slow import, only needed for this plot

    plt.figure()  # summary_plot draws on the current figure; start a clean one
    shap.summary_plot(values.to_numpy(), X, max_display=max_display, show=False, plot_size=(8, 6))
    fig = plt.gcf()
    fig.axes[0].set_xlabel("SHAP value (log-odds): right = more likely to complete")
    fig.axes[0].set_title("What drives the model's completion probability (held-out passes)", fontsize=11)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return fig
