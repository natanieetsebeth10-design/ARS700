
"""
Component A - Step 5b: Trade-off Study

This script runs offline experiments to compare:
- Perception accuracy and robot safety
- Planning resolution, path length, and computation
- Control responsiveness and perception delay

It uses the same planner and robot model as the rest of the project.
It saves a graph and CSV files in the data folder.
"""

import csv
import os

import numpy as np
import matplotlib.pyplot as plt

import Planning as planner
import control_system as ctl

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
os.makedirs(DATA, exist_ok=True)
OBSTACLES_TRUE = list(planner.OBSTACLES)
RES0, N0 = planner.RESOLUTION, planner.GRID_N


def plan_waypoints():
    # Find a path and convert its points to world coordinates.
    grid = planner.build_occupancy_grid()
    s, g = planner.world_to_grid(*planner.START_WORLD), planner.world_to_grid(*planner.GOAL_WORLD)
    path, nodes, rt = planner.astar(grid, s, g)
    if path is None:
        return None, nodes, rt, grid
    wps = [planner.grid_to_world(*c) for c in ctl.simplify_path(grid, path)]
    wps[0], wps[-1] = planner.START_WORLD, planner.GOAL_WORLD
    return wps, nodes, rt, grid


def poly_len(w):
    # Calculate the total length of the path.
    w = np.asarray(w)
    return float(np.sum(np.hypot(*np.diff(w, axis=0).T)))


def clearance_xy(xy):
    # Find the smallest distance between the robot and an obstacle.
    best = np.inf
    for x, y in xy:
        for (cx, cy, hx, hy) in OBSTACLES_TRUE:
            dx, dy = max(abs(x - cx) - hx, 0.0), max(abs(y - cy) - hy, 0.0)
            best = min(best, np.hypot(dx, dy) - planner.ROBOT_RADIUS)
    return best


def sample_polyline(w, step=0.01):
    # Add points along the path to check obstacle clearance.
    pts = []
    for a, b in zip(w[:-1], w[1:]):
        n = max(2, int(np.hypot(b[0] - a[0], b[1] - a[1]) / step))
        for s in np.linspace(0, 1, n):
            pts.append((a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1])))
    return pts


def track_with_offset(wps, offset_xy, gains=ctl.TUNED_GAINS):
    """Run the robot using a position measurement with a fixed offset."""
    ox, oy = offset_xy
    robot = ctl.DiffDriveRobot(wps[0][0], wps[0][1], 0.0)
    pid = ctl.PID(*gains)
    idx, t, true_xy = 1, 0.0, []
    while t < 120.0:
        # The robot uses the offset position to follow the path.
        mx, my = robot.x + ox, robot.y + oy
        dx, dy = wps[idx][0] - mx, wps[idx][1] - my
        dist = np.hypot(dx, dy)
        last = idx == len(wps) - 1
        if dist < (ctl.GOAL_RADIUS if last else ctl.WP_RADIUS):
            if last:
                break
            idx += 1
            continue
        err = ctl.wrap(np.arctan2(dy, dx) - robot.theta)
        robot.step(ctl.V_MAX * max(0.0, np.cos(err)) * min(1.0, dist / 0.15 + 0.3), pid(err))
        true_xy.append((robot.x, robot.y))
        t += ctl.DT
    return np.array(true_xy)


def study_perception():
    # Test how position measurement errors affect tracking and safety.
    wps, *_ = plan_waypoints()
    rows = []
    for sigma in [0.0, 0.01, 0.02, 0.03, 0.05, 0.08]:
        rms, clr = [], []
        for phi in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            xy = track_with_offset(wps, (sigma * np.cos(phi), sigma * np.sin(phi)))
            cte = [min(np.hypot(*(np.array(p) - np.array(q))) for q in sample_polyline(wps, 0.02))
                   for p in xy[::10]]
            rms.append(np.sqrt(np.mean(np.square(cte))))
            clr.append(clearance_xy(xy))
        rows.append((sigma, float(np.mean(rms)), float(np.min(clr))))
    return rows


def study_resolution():
    # Compare grid sizes to see their effect on path length and search effort.
    rows = []
    for res in [0.025, 0.05, 0.1, 0.15, 0.25]:
        planner.RESOLUTION, planner.GRID_N = res, int(round(planner.ARENA_SIZE / res))
        wps, nodes_a, rt_a, grid = plan_waypoints()
        s, g = planner.world_to_grid(*planner.START_WORLD), planner.world_to_grid(*planner.GOAL_WORLD)
        _, nodes_d, rt_d = planner.dijkstra(grid, s, g)
        if wps is None:
            rows.append((res, np.nan, nodes_a, nodes_d, rt_a, np.nan))
            continue
        rows.append((res, poly_len(wps), nodes_a, nodes_d, rt_a, clearance_xy(sample_polyline(wps))))
    # Restore the original grid settings.
    planner.RESOLUTION, planner.GRID_N = RES0, N0
    return rows


LAT_STEP = np.deg2rad(20.0)  # A small step helps show the effect of delay.


def step_with_delay(gains, delay):
    """Test the heading response when feedback is delayed."""
    robot = ctl.DiffDriveRobot(theta=0.0, bias=0.0)
    pid = ctl.PID(*gains)
    nd = int(round(delay / ctl.DT))
    buf = [0.0] * (nd + 1)
    n = int(12.0 / ctl.DT)  # Run long enough to measure slow responses.
    th = np.zeros(n)
    for i in range(n):
        # Use an older heading value to simulate delayed feedback.
        err = ctl.wrap(LAT_STEP - buf[0])
        robot.step(0.0, pid(err))
        buf.append(robot.theta)
        buf.pop(0)
        th[i] = robot.theta
    return np.arange(n) * ctl.DT, th


def scaled(kp, kd_ratio=0.0625):
    # Test proportional and derivative gains without integral gain.
    return (kp, 0.0, kd_ratio * kp)


def study_latency():
    """Find the fastest gains that keep overshoot below 10% and remain stable."""
    delays = [0.0, 0.05, 0.10, 0.15, 0.20]
    curves = {d: step_with_delay(ctl.TUNED_GAINS, d) for d in delays}
    best = []
    for d in delays:
        cand = []
        for kp in [0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48]:
            for kdr in [0.0, 0.02, 0.04, 0.0625]:
                t, th = step_with_delay(scaled(kp, kdr), d)
                rise, os_, settle, sse = ctl.step_metrics(t, th, LAT_STEP)
                stable = th.max() < 1.5 * LAT_STEP and abs(th[-100:].mean() - LAT_STEP) < 0.05
                if stable and os_ < 10.0 and not np.isnan(rise):
                    cand.append((rise, kp, os_))
        best.append((d, *(min(cand) if cand else (np.nan, np.nan, np.nan))))
    return curves, best


def main():
    pa = study_perception()
    pb = study_resolution()
    curves, pd_ = study_latency()

    print("\n(a) PERCEPTION ACCURACY (constant localisation offset, tuned PID)")
    print(f"{'offset (m)':>11}{'RMS true CTE (m)':>20}{'worst clearance (m)':>22}")
    for s, r, c in pa:
        print(f"{s:>11.3f}{r:>20.4f}{c:>22.4f}   {'COLLISION' if c < 0 else ''}")
    print("\n(b) PLANNING RESOLUTION")
    print(f"{'res (m)':>8}{'path len (m)':>14}{'A* nodes':>10}{'Dijk nodes':>12}{'A* time (ms)':>14}{'clearance (m)':>15}")
    for res, L, na, nd, rt, c in pb:
        print(f"{res:>8.3f}{L:>14.3f}{na:>10}{nd:>12}{rt * 1000:>14.2f}{c:>15.4f}")
    print("\n(c) CONTROL RESPONSIVENESS vs PERCEPTION LATENCY (fastest gain with OS<10%)")
    print(f"{'delay (s)':>10}{'best Kp':>9}{'rise time (s)':>15}{'overshoot %':>13}")
    for d, rise, kp, os_ in pd_:
        print(f"{d:>10.2f}{kp:>9}{rise:>15.3f}{os_:>13.1f}")

    # Save the results as CSV files.
    for name, hdr, rows in [("tradeoff_perception.csv", "offset_m,rms_cte_m,worst_clearance_m", pa),
                            ("tradeoff_resolution.csv",
                             "res_m,path_len_m,astar_nodes,dijkstra_nodes,astar_time_s,clearance_m", pb),
                            ("tradeoff_latency.csv", "delay_s,rise_time_s,best_kp,overshoot_pct", pd_)]:
        with open(os.path.join(DATA, name), "w", newline="") as f:
            f.write(hdr + "\n")
            csv.writer(f).writerows(rows)

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # Plot position error and obstacle clearance.
    s = [r[0] * 100 for r in pa]
    ax[0, 0].plot(s, [r[1] * 100 for r in pa], "o-", color="purple", label="RMS cross-track error (cm)")
    ax[0, 0].set_xlabel("Localisation error (cm)")
    ax[0, 0].set_ylabel("RMS cross-track error (cm)", color="purple")
    a2 = ax[0, 0].twinx()
    a2.plot(s, [r[2] * 100 for r in pa], "s--", color="crimson", label="Worst clearance (cm)")
    a2.axhline(0, color="k", lw=1)
    a2.set_ylabel("Worst-case clearance to obstacles (cm)", color="crimson")
    ax[0, 0].set_title("(a) Perception accuracy vs safety\n(below the black line = collision)")

    # Compare path length and search effort at each grid size.
    res = [r[0] * 100 for r in pb]
    ax[0, 1].plot(res, [r[1] for r in pb], "o-", color="tab:blue")
    ax[0, 1].set_xlabel("Grid resolution (cm/cell)")
    ax[0, 1].set_ylabel("Planned path length (m)", color="tab:blue")
    b2 = ax[0, 1].twinx()
    b2.semilogy(res, [r[2] for r in pb], "s--", color="tab:orange", label="A*")
    b2.semilogy(res, [r[3] for r in pb], "^:", color="tab:green", label="Dijkstra")
    b2.set_ylabel("Nodes explored (log)")
    b2.legend(loc="center right", fontsize=8)
    ax[0, 1].set_title("(b) Planning resolution: optimality vs computation")

    # Show how perception delay affects the heading response.
    for d, (t, th) in curves.items():
        ax[1, 0].plot(t, np.degrees(th), label=f"latency {d * 1000:.0f} ms")
    ax[1, 0].axhline(20, color="k", ls="--", lw=1)
    ax[1, 0].set_ylim(0, 50)
    ax[1, 0].set_xlim(0, 8)
    ax[1, 0].set_xlabel("Time (s)")
    ax[1, 0].set_ylabel("Heading (deg)")
    ax[1, 0].set_title("(c) Same PID (Kp=8, Ki=6, Kd=0.5), 20° step, under perception latency")
    ax[1, 0].legend(fontsize=8)

    # Plot the fastest response found for each delay.
    ax[1, 1].plot([r[0] * 1000 for r in pd_], [r[1] for r in pd_], "o-", color="tab:red")
    ax[1, 1].set_xlabel("Perception latency (ms)")
    ax[1, 1].set_ylabel("Fastest achievable rise time, 10-90% (s)")
    for d, st, kp, _ in pd_:
        ax[1, 1].annotate(f"Kp={kp}", (d * 1000, st), textcoords="offset points", xytext=(5, 6), fontsize=8)
    ax[1, 1].set_title("(d) Latency forces lower gains -> slower response\n(fastest gain with overshoot < 10%)")
    for a in ax.flat:
        a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA, "tradeoff_study.png"), dpi=150)
    print("\nSaved: data/tradeoff_study.png and tradeoff_*.csv")
    plt.show()


if __name__ == "__main__":
    main()
