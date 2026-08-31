"""Generate paper/delphi_process_flowchart.png in a clean architecture-diagram style."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.lines import Line2D

FONT = "DejaVu Sans"

COLORS = {
    "init":     dict(face="#DCE6F7", edge="#3B5FA0"),
    "round1":   dict(face="#DDEFE1", edge="#2E8B57"),
    "process":  dict(face="#E6DEF2", edge="#6A4C93"),
    "round2":   dict(face="#FCE6D6", edge="#D9622B"),
    "round3":   dict(face="#FCE6D6", edge="#D9622B"),
    "terminal": dict(face="#DDEFE1", edge="#2E8B57"),
    "final":    dict(face="#DCE6F7", edge="#3B5FA0"),
    "decision": dict(face="#F3EDF9", edge="#6A4C93"),
}
GREEN = "#2E8B57"
RED = "#C0392B"
DARK = "#1A1A2E"


def box(ax, x, y, w, h, title, body, style, title_color=None, title_size=12, body_size=9.6):
    c = COLORS[style]
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=2.2, edgecolor=c["edge"], facecolor=c["face"], zorder=2,
    ))
    tc = title_color or c["edge"]
    n_title_lines = title.count("\n") + 1 if title else 0
    title_top = y + h - 0.28
    ax.text(x + w / 2, title_top, title, ha="center", va="top",
             fontsize=title_size, fontweight="bold", color=tc, family=FONT,
             linespacing=1.35, zorder=3)
    body_top = title_top - n_title_lines * (title_size / 72 * 1.35) - 0.22
    ax.text(x + w / 2, body_top, body, ha="center", va="top",
             fontsize=body_size, color=DARK, family=FONT, linespacing=1.55, zorder=3)
    return x, y, w, h


def circle_node(ax, x, y, r, title, edge):
    ax.add_patch(Circle((x, y), r, linewidth=2.4, edgecolor=edge,
                         facecolor=COLORS["terminal"]["face"], zorder=2))
    ax.text(x, y, title, ha="center", va="center", fontsize=11.5, fontweight="bold",
             color=edge, family=FONT, linespacing=1.3, zorder=3)
    return x, y, r


def arrow(ax, p0, p1, color=DARK, lw=2.0, connectionstyle="arc3,rad=0.0"):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=16, linewidth=lw,
        color=color, connectionstyle=connectionstyle, zorder=1,
    ))


def edge_label(ax, x, y, text, color):
    ax.text(x, y, text, ha="center", va="center", fontsize=10.5, fontweight="bold",
             color=color, family=FONT, zorder=4,
             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.9))


fig, ax = plt.subplots(figsize=(16.5, 6.3))
ax.set_xlim(0, 17.8)
ax.set_ylim(-0.2, 6.1)
ax.axis("off")

ax.text(8.9, 5.9, "Delphi-style consensus workflow for multimodal 3D evaluation",
        ha="center", va="top", fontsize=18, fontweight="bold", color=DARK, family=FONT)

# --- Row 1: sequential phase boxes + decision + terminal circle -------------
row_y, row_h = 2.3, 2.35
mid_y = row_y + row_h / 2

p1 = box(ax, 0.3, row_y, 2.6, row_h, "PHASE 1\nINITIALIZATION",
         "Inputs: multi-view\nrenderings, text prompt,\ncriteria, VQA questions", "init")

p2 = box(ax, 3.25, row_y, 2.6, row_h, "PHASE 2 · ROUND 1",
         "Independent rating\nClaude, GPT-4.1, Gemini\nscore 0–9 + rationale", "round1")

p3 = box(ax, 6.2, row_y, 2.6, row_h, "PHASE 3\nINTER-ROUND\nPROCESSING",
         "Median & IQR computed;\nanonymized dissent\nsummary generated", "process")

p4 = box(ax, 9.15, row_y, 2.6, row_h, "PHASE 4 · ROUND 2",
         "Informed re-rating—judges\nrevise using median,\nIQR, dissent summary", "round2")

dc = box(ax, 12.1, 2.15, 2.35, 2.65, "CONVERGENCE\nCHECK",
         "SD < 1.0 on\n≥4 of 5 criteria", "decision", title_color=DARK,
         title_size=12.5, body_size=10.5)

circ_x, circ_y, circ_r = circle_node(ax, 16.35, mid_y, 1.15, "ITERATION\nTERMINATED", GREEN)

chain = [p1, p2, p3, p4, dc]
for (x0, y0, w0, h0), (x1, y1, w1, h1) in zip(chain[:-1], chain[1:]):
    arrow(ax, (x0 + w0, mid_y), (x1, mid_y))

# Yes -> terminated (green)
arrow(ax, (dc[0] + dc[2], dc[1] + dc[3] - 0.35), (circ_x - circ_r * 0.85, circ_y + circ_r * 0.5),
      color=GREEN, lw=2.3, connectionstyle="arc3,rad=-0.2")
edge_label(ax, 15.15, 4.95, "Yes", GREEN)

# --- Row 2: Round 3 + final consensus matrix --------------------------------
r3 = box(ax, 11.4, 0.2, 2.7, 1.7, "PHASE 5 · ROUND 3",
         "Final score revision\nusing updated R2\nstatistics", "round3", title_size=12, body_size=9.3)

# No -> round 3 (red)
arrow(ax, (dc[0] + dc[2] / 2, dc[1]), (r3[0] + r3[2] * 0.85, r3[1] + r3[3]),
      color=RED, lw=2.3, connectionstyle="arc3,rad=0.05")
edge_label(ax, 14.05, 2.1, "No", RED)

fm = box(ax, 4.9, 0.2, 5.3, 1.5, "FINAL CONSENSUS MATRIX",
         "Calibrated per-criterion outputs, mapped forward\ninto the Item Response Theory pipeline",
         "final", title_size=13, body_size=10)

# terminated -> final matrix (green)
arrow(ax, (circ_x, circ_y - circ_r), (fm[0] + fm[2] * 0.95, fm[1] + fm[3]),
      color=GREEN, lw=2.3, connectionstyle="arc3,rad=0.28")
# round3 -> final matrix (red)
arrow(ax, (r3[0], r3[1] + r3[3] * 0.4), (fm[0] + fm[2], fm[1] + fm[3] * 0.55),
      color=RED, lw=2.3, connectionstyle="arc3,rad=-0.1")

# --- Legend -------------------------------------------------------------------
leg_x, leg_y = 13.3, 5.55
ax.add_line(Line2D([leg_x, leg_x + 0.55], [leg_y, leg_y], color=GREEN, lw=2.6))
ax.text(leg_x + 0.7, leg_y, "Resolved / converged path", fontsize=10, color=DARK,
        va="center", family=FONT)
ax.add_line(Line2D([leg_x, leg_x + 0.55], [leg_y - 0.32, leg_y - 0.32], color=RED, lw=2.6))
ax.text(leg_x + 0.7, leg_y - 0.32, "Unresolved / continues", fontsize=10, color=DARK,
        va="center", family=FONT)

plt.savefig("paper/delphi_process_flowchart.png",
            dpi=220, bbox_inches="tight", facecolor="white")
print("done")
