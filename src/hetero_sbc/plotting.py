from __future__ import annotations

from pathlib import Path

from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
import numpy as np

from .config import ExperimentResult


def plot_trajectories(result: ExperimentResult, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    positions = result.positions
    radii = np.asarray(result.metadata["radii"], dtype=float)
    for i in range(positions.shape[1]):
        ax.plot(positions[:, i, 0], positions[:, i, 1], linewidth=1.5)
        ax.scatter(positions[0, i, 0], positions[0, i, 1], marker="s", s=40, color=ax.lines[-1].get_color())
        ax.scatter(positions[-1, i, 0], positions[-1, i, 1], marker="x", s=60, color=ax.lines[-1].get_color())
        circle = Circle((positions[-1, i, 0], positions[-1, i, 1]), radii[i], fill=False, alpha=0.25)
        ax.add_patch(circle)
    ax.set_title(f"{result.name} | {result.controller}")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_time_series(result: ExperimentResult, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(result.time, result.clearance_history, label="Clearance margin")
    axes[0].axhline(0.0, color="tab:red", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Clearance [m]")
    axes[0].grid(True, alpha=0.2)
    axes[1].plot(result.time, result.cbf_history, label="Min CBF value", color="tab:orange")
    axes[1].axhline(0.0, color="tab:red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("CBF")
    axes[1].grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_scalability(results: list[ExperimentResult], output_path: Path) -> None:
    agent_counts = [r.positions.shape[1] for r in results]
    mean_qp = [r.summary["p95_qp_ms"] for r in results]
    min_clearance = [r.summary["min_clearance"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(agent_counts, mean_qp, marker="o")
    axes[0].set_xlabel("Number of robots")
    axes[0].set_ylabel("P95 QP solve time [ms]")
    axes[0].grid(True, alpha=0.2)
    axes[1].plot(agent_counts, min_clearance, marker="o", color="tab:green")
    axes[1].axhline(0.0, color="tab:red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Number of robots")
    axes[1].set_ylabel("Minimum clearance [m]")
    axes[1].grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_scalability_comparison(results: list[ExperimentResult], output_path: Path) -> None:
    if not results:
        return
    controller_order = []
    grouped: dict[str, list[ExperimentResult]] = {}
    for result in results:
        if result.controller not in grouped:
            grouped[result.controller] = []
            controller_order.append(result.controller)
        grouped[result.controller].append(result)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for controller in controller_order:
        controller_results = sorted(grouped[controller], key=lambda item: item.positions.shape[1])
        agent_counts = [r.positions.shape[1] for r in controller_results]
        mean_qp = [r.summary["p95_qp_ms"] for r in controller_results]
        min_clearance = [r.summary["min_clearance"] for r in controller_results]
        axes[0].plot(agent_counts, mean_qp, marker="o", label=controller)
        axes[1].plot(agent_counts, min_clearance, marker="o", label=controller)

    axes[0].set_xlabel("Number of robots")
    axes[0].set_ylabel("P95 QP solve time [ms]")
    axes[0].grid(True, alpha=0.2)
    axes[0].legend(fontsize=8)
    axes[1].axhline(0.0, color="tab:red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Number of robots")
    axes[1].set_ylabel("Minimum clearance [m]")
    axes[1].grid(True, alpha=0.2)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_outcome_breakdown(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return
    labels = []
    success_runs = []
    deadlock_runs = []
    collision_runs = []
    incomplete_runs = []
    for row in rows:
        controller = row["controller"].replace("_heterogeneous_barrier", "_het")
        labels.append(f'{row["experiment"]}\n{controller}')
        success_runs.append(row["success_runs"])
        deadlock_runs.append(row["deadlock_runs"])
        collision_runs.append(row["collision_runs"])
        incomplete_runs.append(row["incomplete_runs"])

    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(rows)), 4.5))
    ax.bar(x, success_runs, label="success")
    ax.bar(x, deadlock_runs, bottom=success_runs, label="deadlock")
    ax.bar(
        x,
        collision_runs,
        bottom=np.asarray(success_runs) + np.asarray(deadlock_runs),
        label="collision",
    )
    ax.bar(
        x,
        incomplete_runs,
        bottom=np.asarray(success_runs) + np.asarray(deadlock_runs) + np.asarray(collision_runs),
        label="incomplete",
    )
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("Runs")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_sensitivity_grid(
    ds_values: list[float],
    gamma_values: list[float],
    metric_grid: np.ndarray,
    label: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(metric_grid, origin="lower", aspect="auto")
    ax.set_xticks(range(len(gamma_values)), [f"{v:.1f}" for v in gamma_values])
    ax.set_yticks(range(len(ds_values)), [f"{v:.2f}" for v in ds_values])
    ax.set_xlabel("gamma")
    ax.set_ylabel("Safety buffer D_s [m]")
    ax.set_title(label)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def animate_robotarium_style(
    result: ExperimentResult,
    goals: np.ndarray,
    output_path: Path,
    fps: int = 12,
    trail_length: int = 20,
    frame_skip: int = 1,
) -> None:
    positions = result.positions[::frame_skip]
    velocities = result.velocities[::frame_skip]
    time = result.time[: result.positions.shape[0] : frame_skip]
    radii = np.asarray(result.metadata["radii"], dtype=float)
    clearance = result.clearance_history[: result.positions.shape[0] : frame_skip]

    colors = plt.cm.tab10(np.linspace(0.0, 1.0, positions.shape[1], endpoint=False))
    extent = float(np.max(np.abs(np.concatenate([positions.reshape(-1, 2), goals], axis=0))) + np.max(radii) + 0.5)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.set_facecolor("#f5f1e8")
    arena = Rectangle(
        (-extent, -extent),
        2.0 * extent,
        2.0 * extent,
        linewidth=2.0,
        edgecolor="#3d3d3d",
        facecolor="#fbfaf5",
    )
    ax.add_patch(arena)
    ax.axhline(0.0, color="#d8d1c3", linestyle="--", linewidth=1.0)
    ax.axvline(0.0, color="#d8d1c3", linestyle="--", linewidth=1.0)

    goal_markers = []
    robot_patches = []
    heading_arrows: list[FancyArrowPatch] = []
    trails = []
    labels = []
    for idx, color in enumerate(colors):
        goal = ax.scatter(
            goals[idx, 0],
            goals[idx, 1],
            marker="X",
            s=80,
            color=color,
            edgecolors="#202020",
            linewidths=0.6,
            alpha=0.9,
            zorder=2,
        )
        goal_markers.append(goal)
        trail, = ax.plot([], [], color=color, linewidth=1.4, alpha=0.55, zorder=2)
        trails.append(trail)
        robot = Circle((positions[0, idx, 0], positions[0, idx, 1]), radii[idx], facecolor=color, edgecolor="#202020", linewidth=1.0, zorder=3)
        ax.add_patch(robot)
        robot_patches.append(robot)
        arrow = FancyArrowPatch((0.0, 0.0), (0.0, 0.0), arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color="#202020", zorder=4)
        ax.add_patch(arrow)
        heading_arrows.append(arrow)
        label = ax.text(positions[0, idx, 0], positions[0, idx, 1], str(idx + 1), ha="center", va="center", fontsize=8, color="white", zorder=5)
        labels.append(label)

    status_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#999999", "alpha": 0.9, "pad": 6},
    )

    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{result.name} | {result.controller}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(False)

    def _update(frame_idx: int) -> list:
        artists: list = [status_text]
        for idx in range(positions.shape[1]):
            center = positions[frame_idx, idx]
            velocity = velocities[frame_idx, idx]
            robot_patches[idx].center = tuple(center)
            labels[idx].set_position(tuple(center))
            heading = velocity.copy()
            speed = float(np.linalg.norm(heading))
            if speed > 1e-9:
                heading = heading / speed
            else:
                heading = np.array([1.0, 0.0], dtype=float)
            arrow_start = center + heading * radii[idx] * 0.35
            arrow_end = center + heading * (radii[idx] + 0.18)
            heading_arrows[idx].set_positions(tuple(arrow_start), tuple(arrow_end))

            trail_start = max(0, frame_idx - trail_length)
            trails[idx].set_data(positions[trail_start : frame_idx + 1, idx, 0], positions[trail_start : frame_idx + 1, idx, 1])
            artists.extend([robot_patches[idx], heading_arrows[idx], trails[idx], labels[idx], goal_markers[idx]])

        status_text.set_text(
            "\n".join(
                [
                    f"t = {time[frame_idx]:.2f} s",
                    f"min clearance = {clearance[frame_idx]:.3f} m",
                    f"controller = {result.controller}",
                ]
            )
        )
        return artists

    anim = animation.FuncAnimation(fig, _update, frames=positions.shape[0], interval=1000 / fps, blit=False)
    writer = animation.PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=120)
    plt.close(fig)
