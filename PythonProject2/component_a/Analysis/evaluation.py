
"""
Component A - Step 5: System Integration and Evaluation

Run this script in PyCharm after running the Webots simulation.
It reads the navigation logs and creates plots and performance tables
for each run. If there are multiple runs, it also compares them.

The output includes the robot's path, errors over time, and performance
measurements for the project report.
"""

import csv
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap

import Planning as planner
from control_system import simplify_path

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

# This obstacle exists in the Webots world but is missing from the map.
# Keep these coordinates matched to obstacle_5_hidden in the world file.
# Use [] if you removed that obstacle.
HIDDEN_TRUE = [(-0.38, -0.34, 0.10, 0.10)]


# Load the saved navigation data.
def load(run_dir):
    rows = list(csv.DictReader(open(os.path.join(run_dir, "nav_log.csv"))))
    cols = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    num = {k: v.astype(float) for k, v in cols.items() if k != "state"}
    num["state"] = cols["state"]

    plans = json.load(open(os.path.join(run_dir, "nav_plans.json")))
    summary = json.load(open(os.path.join(run_dir, "nav_summary.json")))
    g0 = np.load(os.path.join(run_dir, "nav_grid_initial.npy"))
    g1 = np.load(os.path.join(run_dir, "nav_grid_final.npy"))

    return num, plans, summary, g0, g1


# Calculate the total length of a path.
def polyline_length(pts):
    pts = np.asarray(pts)
    return float(np.sum(np.hypot(*np.diff(pts, axis=0).T)))


# Find the shortest distance from a point to a path.
def dist_to_polyline(p, poly):
    best = np.inf

    for i in range(len(poly) - 1):
        a, b = np.asarray(poly[i]), np.asarray(poly[i + 1])
        ab = b - a
        den = float(np.dot(ab, ab))

        tt = 0.0 if den == 0 else np.clip(
            np.dot(p - a, ab) / den, 0, 1
        )

        best = min(
            best,
            float(np.hypot(*(p - (a + tt * ab))))
        )

    return best


# Find the smallest distance between the robot and any obstacle.
def min_clearance(xy, obstacles):
    best = np.inf

    for x, y in xy:
        for (cx, cy, hx, hy) in obstacles:
            dx = max(abs(x - cx) - hx, 0.0)
            dy = max(abs(y - cy) - hy, 0.0)

            best = min(
                best,
                np.hypot(dx, dy) - planner.ROBOT_RADIUS
            )

    return best


# Find the shortest path when all obstacles are known.
def ideal_path_length():
    saved = list(planner.OBSTACLES)

    # Temporarily include the hidden obstacle in the map.
    planner.OBSTACLES[:] = saved + HIDDEN_TRUE

    grid = planner.build_occupancy_grid()

    s = planner.world_to_grid(*planner.START_WORLD)
    g = planner.world_to_grid(*planner.GOAL_WORLD)

    path, _, _ = planner.astar(grid, s, g)

    # Restore the original map.
    planner.OBSTACLES[:] = saved

    wps = [
        planner.grid_to_world(*c)
        for c in simplify_path(grid, path)
    ]

    wps[0], wps[-1] = planner.START_WORLD, planner.GOAL_WORLD

    return polyline_length(wps)


# Calculate the performance of one navigation run.
def evaluate(run_dir):
    d, plans, summary, g0, g1 = load(run_dir)

    t = d["t"]
    true_xy = np.c_[d["true_x"], d["true_y"]]
    est_xy = np.c_[d["est_x"], d["est_y"]]
    plan_id = d["plan_id"].astype(int)

    true_obstacles = list(planner.OBSTACLES) + HIDDEN_TRUE

    # Calculate errors at each recorded time step.
    cte = np.array([
        dist_to_polyline(
            true_xy[i],
            plans[plan_id[i]]["waypoints"]
        )
        for i in range(len(t))
    ])

    # Compare the estimated position with the actual position.
    loc_err = np.hypot(*(true_xy - est_xy).T)

    # Convert the heading error from radians to degrees.
    head_err_deg = np.degrees(np.abs(d["heading_err"]))

    # Calculate the overall performance measurements.
    actual_len = polyline_length(true_xy)
    init_planned = polyline_length(plans[0]["waypoints"])
    ideal_len = ideal_path_length()
    clearance = min_clearance(true_xy, true_obstacles)
    t_goal = float(t[-1])
    est_state = d["state"]

    metrics = [
        ("Goal reached (controller's belief)",
         "Yes" if summary["reached"] else "No"),
        ("Localisation source",
         summary.get("pose_source", "odometry")),
        ("Final distance to goal (true, m)",
         f"{summary['final_goal_error_m']:.3f}"),
        ("Time to goal (s)", f"{t_goal:.1f}"),
        ("Initial planned length (known map, m)",
         f"{init_planned:.3f}"),
        ("Optimal length on true map (A*, m)",
         f"{ideal_len:.3f}"),
        ("Actual distance travelled (m)",
         f"{actual_len:.3f}"),
        ("Path efficiency (optimal / actual)",
         f"{ideal_len / actual_len:.3f}"),
        ("Mean speed (m/s)", f"{actual_len / t_goal:.3f}"),
        ("Replans triggered by perception",
         f"{summary['replans']}"),
        ("Mean cross-track error (m)", f"{cte.mean():.4f}"),
        ("RMS cross-track error (m)",
         f"{np.sqrt(np.mean(cte ** 2)):.4f}"),
        ("Max cross-track error (m)", f"{cte.max():.4f}"),
        ("Mean |heading error| (deg)",
         f"{head_err_deg.mean():.2f}"),
        ("Mean odometry error vs truth (m)",
         f"{loc_err.mean():.4f}"),
        ("Final odometry error vs truth (m)",
         f"{loc_err[-1]:.4f}"),
        ("Min clearance to obstacles (m)",
         f"{clearance:.4f}"),
        ("Collision (clearance < 0)",
         "YES" if clearance < 0 else "No"),
        ("Time in SLOW / ESTOP (s)",
         f"{np.sum(est_state == 'SLOW') * 0.016:.1f} / "
         f"{np.sum(est_state == 'ESTOP') * 0.016:.1f}"),
    ]

    # Print the performance results.
    print("=" * 66)
    print(
        f"COMPONENT A - INTEGRATED NAVIGATION: PERFORMANCE EVALUATION "
        f"[{os.path.basename(run_dir)}]"
    )
    print("=" * 66)

    for k, v in metrics:
        print(f"{k:<46}{v:>18}")

    # Save the results in a CSV file.
    with open(
        os.path.join(run_dir, "nav_performance_table.csv"),
        "w",
        newline=""
    ) as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows(metrics)

    # Plot the planned path and the path the robot actually followed.
    fig, ax = plt.subplots(figsize=(8.5, 8))

    cmap = ListedColormap([
        "#ffffff",
        "#d0d0d0",
        "#f4b6f4"
    ])

    # Show the original map and cells added after obstacle detection.
    show = g0.astype(int) + 2 * (
        (g1 == 1) & (g0 == 0)
    ).astype(int)

    ax.imshow(
        show.T,
        origin="lower",
        cmap=cmap,
        vmin=0,
        vmax=2,
        extent=[-1.5, 1.5, -1.5, 1.5],
        alpha=0.9
    )

    # Draw the obstacles already known to the planner.
    for (cx, cy, hx, hy) in planner.OBSTACLES:
        ax.add_patch(
            Rectangle(
                (cx - hx, cy - hy),
                2 * hx,
                2 * hy,
                fill=False,
                ec="k",
                lw=1.5
            )
        )

    # Show the obstacle that was missing from the original map.
    for (cx, cy, hx, hy) in HIDDEN_TRUE:
        ax.add_patch(
            Rectangle(
                (cx - hx, cy - hy),
                2 * hx,
                2 * hy,
                fill=False,
                ec="m",
                lw=2,
                ls="--",
                label="Hidden obstacle (true position)"
            )
        )

    # Draw the original plan and any new plans.
    for i, p in enumerate(plans):
        wx, wy = zip(*p["waypoints"])

        ax.plot(
            wx,
            wy,
            "--o",
            ms=5,
            lw=1.8,
            color="tab:blue" if i == 0 else "tab:green",
            label=(
                "Initial plan (known map)"
                if i == 0
                else f"Replan {i} (t={p['t']:.1f}s)"
            )
        )

    # Compare the estimated path with the actual path.
    ax.plot(
        est_xy[:, 0],
        est_xy[:, 1],
        ":",
        color="orange",
        lw=1.5,
        label="Odometry estimate"
    )

    ax.plot(
        true_xy[:, 0],
        true_xy[:, 1],
        "-",
        color="crimson",
        lw=2.2,
        label="Actual path (true pose)"
    )

    ax.plot(
        *planner.START_WORLD,
        "go",
        ms=11,
        label="Start"
    )

    ax.plot(
        *planner.GOAL_WORLD,
        "b*",
        ms=16,
        label="Goal"
    )

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    ax.set_title(
        f"Planned vs actual trajectory "
        f"[{summary.get('pose_source', '')} localisation]\n"
        "grey = inflated known map, pink = cells added by perception",
        fontsize=9
    )

    ax.legend(loc="lower right", fontsize=7)

    plt.tight_layout()
    plt.savefig(
        os.path.join(run_dir, "nav_planned_vs_actual.png"),
        dpi=150
    )

    # Plot the navigation errors and control commands over time.
    fig, axes = plt.subplots(
        2, 2,
        figsize=(13, 7.5),
        sharex=True
    )

    # Get the times when the robot planned a new path.
    replan_t = [p["t"] for p in plans[1:]]

    def mark(ax):
        """Mark the times when a new path was planned."""

        for k, rt in enumerate(replan_t):
            ax.axvline(
                rt,
                color="green",
                ls="--",
                lw=1,
                label="Replan" if k == 0 else None
            )

    # Show how far the robot moved away from its planned path.
    axes[0, 0].plot(t, cte, color="purple")
    axes[0, 0].set_ylabel("Cross-track error (m)")
    axes[0, 0].set_title("Tracking accuracy (true pose vs active plan)")

    # Show the robot's heading error.
    axes[0, 1].plot(t, head_err_deg, color="orange")
    axes[0, 1].set_ylabel("|Heading error| (deg)")
    axes[0, 1].set_title("PID heading error")

    # Show the forward and turning commands.
    axes[1, 0].plot(
        t,
        d["v_cmd"],
        label="v command (m/s)"
    )

    ax2 = axes[1, 0].twinx()

    ax2.plot(
        t,
        d["w_cmd"],
        color="tab:red",
        alpha=0.7,
        label="w command (rad/s)"
    )

    axes[1, 0].set_ylabel("v (m/s)")
    ax2.set_ylabel("w (rad/s)")
    axes[1, 0].set_title("Control effort")

    # Show the difference between estimated and actual position.
    axes[1, 1].plot(t, loc_err, color="tab:brown")
    axes[1, 1].set_ylabel("Odometry error (m)")
    axes[1, 1].set_title("Localisation accuracy: |odometry - truth|")

    # Add grid lines and mark each replan on the graphs.
    for a in axes.flat:
        a.grid(alpha=0.3)
        mark(a)

    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 1].set_xlabel("Time (s)")

    if replan_t:
        axes[0, 0].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(
        os.path.join(run_dir, "nav_error_analysis.png"),
        dpi=150
    )

    # Create an image of the performance table.
    fig, ax = plt.subplots(
        figsize=(8.5, 0.34 * len(metrics) + 0.8)
    )

    ax.axis("off")

    tb = ax.table(
        cellText=[[k, v] for k, v in metrics],
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
        colWidths=[0.7, 0.3]
    )

    tb.auto_set_font_size(False)
    tb.set_fontsize(9)
    tb.scale(1, 1.3)

    # Format the table headings.
    for (r, c), cell in tb.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#dddddd")

    ax.set_title(
        "Integrated navigation - performance evaluation",
        fontsize=11
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(run_dir, "nav_performance_table.png"),
        dpi=150
    )

    print(
        f"\nSaved to {os.path.basename(run_dir)}/: "
        "nav_planned_vs_actual.png, "
        "nav_error_analysis.png, "
        "nav_performance_table.png/.csv\n"
    )

    return metrics


# Find and evaluate all available navigation runs.
def main():
    runs = sorted(
        r for r in os.listdir(DATA)
        if r.startswith("run_")
        and os.path.exists(os.path.join(DATA, r, "nav_log.csv"))
    )

    if not runs:
        print(
            "No run_*/nav_log.csv found in data/. "
            "Run the Webots simulation first."
        )
        return

    results = {
        r: evaluate(os.path.join(DATA, r))
        for r in runs
    }

    # Compare the runs if more than one is available.
    if len(runs) > 1:
        names = [k for k, _ in results[runs[0]]]

        table = [
            [n] + [dict(results[r]).get(n, "") for r in runs]
            for n in names
        ]

        # Save the comparison results as a CSV file.
        with open(
            os.path.join(DATA, "nav_run_comparison.csv"),
            "w",
            newline=""
        ) as f:
            w = csv.writer(f)
            w.writerow(["metric"] + runs)
            w.writerows(table)

        # Create a table comparing the runs.
        fig, ax = plt.subplots(
            figsize=(10, 0.34 * len(table) + 0.9)
        )

        ax.axis("off")

        tb = ax.table(
            cellText=table,
            colLabels=["Metric"] + [
                r.replace("run_", "") for r in runs
            ],
            loc="center",
            cellLoc="left",
            colWidths=[0.5] + [0.5 / len(runs)] * len(runs)
        )

        tb.auto_set_font_size(False)
        tb.set_fontsize(9)
        tb.scale(1, 1.3)

        # Format the table headings.
        for (r, c), cell in tb.get_celld().items():
            if r == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#dddddd")

        ax.set_title(
            "Localisation source comparison "
            "(same world, same controller)",
            fontsize=11
        )

        plt.tight_layout()
        plt.savefig(
            os.path.join(DATA, "nav_run_comparison.png"),
            dpi=150
        )

        print("Saved data/nav_run_comparison.csv/.png")

    plt.show()


if __name__ == "__main__":
    main()
