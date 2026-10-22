"""
swarm_supervisor.py - independent ground-truth logger and stop condition (Component B).

The supervisor reads the TRUE positions of the 4 e-pucks (the robots never see this
data) and records:
  * trajectories                         -> supervisor_log.csv
  * sectors completed over time          (a sector is completed once robots have
                                          stood inside it for SECTOR_WORK[s] seconds)
  * safety: robot-robot and robot-obstacle contacts, minimum separation
  * mission completion time              -> summary.json
When every sector is complete (or after TIMEOUT s) it writes the summary and pauses
the simulation.
"""
import csv
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "swarm_agent"))
import swarm_core as sc  # noqa: E402

TIMEOUT = 260.0
SETTLE = 1.0                      # seconds to keep logging after completion
DATA_DIR = os.path.join(HERE, "..", "..", "data")


def run(mode=None, robot=None):
    if robot is None:
        from controller import Supervisor
        robot = Supervisor()
    mode = mode or (sys.argv[1] if len(sys.argv) > 1 else "distributed")
    ts = int(robot.getBasicTimeStep())
    nodes = [robot.getFromDef(f"EPUCK{i}") for i in range(sc.N_AGENTS)]
    centres = sc.sector_centres()
    n_sec = len(centres)
    dwell = [0.0] * n_sec
    done_at = [None] * n_sec
    run_dir = os.path.join(DATA_DIR, f"swarm_run_{mode}")
    os.makedirs(run_dir, exist_ok=True)
    fh = open(os.path.join(run_dir, "supervisor_log.csv"), "w", newline="")
    wr = csv.writer(fh)
    wr.writerow(["t"] + [f"{c}{i}" for i in range(sc.N_AGENTS) for c in ("x", "y")] + ["sectors_done"])
    dt = ts / 1000.0
    min_sep = 9.0
    robot_contacts = obstacle_contacts = 0
    prev_rc = [False] * (sc.N_AGENTS * (sc.N_AGENTS - 1) // 2)
    prev_oc = [False] * (sc.N_AGENTS * len(sc.OBSTACLES))
    finish_t, step_i, written = None, 0, False

    while robot.step(ts) != -1:
        t = robot.getTime()
        step_i += 1
        pos = [n.getPosition() for n in nodes]
        k = 0
        for i in range(sc.N_AGENTS):
            for j in range(i + 1, sc.N_AGENTS):
                d = math.hypot(pos[i][0] - pos[j][0], pos[i][1] - pos[j][1])
                min_sep = min(min_sep, d)
                touching = d < 2 * sc.ROBOT_RADIUS
                if touching and not prev_rc[k]:
                    robot_contacts += 1
                prev_rc[k] = touching
                k += 1
        k = 0
        for i in range(sc.N_AGENTS):
            for (ox, oy, h) in sc.OBSTACLES:
                dx = max(abs(pos[i][0] - ox) - h, 0.0)
                dy = max(abs(pos[i][1] - oy) - h, 0.0)
                touching = math.hypot(dx, dy) < sc.ROBOT_RADIUS
                if touching and not prev_oc[k]:
                    obstacle_contacts += 1
                prev_oc[k] = touching
                k += 1
        for s, (cx, cy) in enumerate(centres):
            if done_at[s] is None and any(math.hypot(p[0] - cx, p[1] - cy) < sc.VISIT_RADIUS for p in pos):
                dwell[s] += dt
                if dwell[s] >= sc.SECTOR_WORK[s] - 0.1:
                    done_at[s] = t
        n_done = sum(d is not None for d in done_at)
        if step_i % 5 == 0:
            wr.writerow([f"{t:.3f}"] + [f"{p[k]:.4f}" for p in pos for k in (0, 1)] + [n_done])
        if finish_t is None and (n_done == n_sec or t > TIMEOUT):
            finish_t = t
            print(f"[supervisor] {'MISSION COMPLETE' if n_done == n_sec else 'TIMEOUT'} at t={t:.1f}s "
                  f"({n_done}/{n_sec} sectors)")
        if finish_t is not None and t >= finish_t + SETTLE and not written:
            written = True
            fh.flush()
            summary = dict(mode=mode, complete=n_done == n_sec, mission_time_s=finish_t,
                           sectors_done=n_done, sector_done_times=done_at,
                           robot_robot_contacts=robot_contacts, robot_obstacle_contacts=obstacle_contacts,
                           min_robot_separation_m=min_sep)
            with open(os.path.join(run_dir, "summary.json"), "w") as sf:
                json.dump(summary, sf, indent=2)
            print(f"[supervisor] summary written to data/swarm_run_{mode}/summary.json | "
                  f"robot contacts {robot_contacts}, obstacle contacts {obstacle_contacts}, "
                  f"min separation {min_sep * 100:.1f} cm")
            try:
                robot.simulationSetMode(robot.SIMULATION_MODE_PAUSE)
            except Exception:
                pass
    fh.close()


if __name__ == "__main__":
    run()