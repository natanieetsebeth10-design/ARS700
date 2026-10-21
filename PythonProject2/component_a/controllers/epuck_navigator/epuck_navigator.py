"""
Component A - Step 4: Integrated Autonomous Navigation

This Webots controller makes the e-puck detect obstacles, plan a path,
and move toward the goal.

Required files:
- Analysis/Planning.py
- Analysis/control_system.py

Perception:
The camera detects coloured obstacles and estimates their direction.
IR sensors estimate distance and detect nearby obstacles.
Both sensors are combined to improve obstacle detection.

Planning:
A* finds a path to the goal.
If a new obstacle is detected, the map is updated and a new path is planned.

Control:
PID control turns the robot toward each waypoint.
Wheel encoders estimate the robot's position.

The controller saves navigation logs, plans, maps and camera images.
"""

import os
import sys
import csv
import json
import math

os.environ.setdefault("MPLBACKEND", "Agg")  # Prevent GUI windows from opening.

import numpy as np
import cv2
from controller import Supervisor

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", ".."))
for _folder in ("python_analysis", "Analysis"):
    sys.path.insert(0, os.path.join(PROJECT, _folder))

DATA_DIR = os.path.join(PROJECT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

try:
    import day2_planning as planner
    from day3_control import (PID, simplify_path, wrap, TUNED_GAINS,
                              WHEEL_R, AXLE_L, MAX_WHEEL_W, V_MAX)
except ImportError:
    import Planning as planner
    from control_system import (PID, simplify_path, wrap, TUNED_GAINS,
                                WHEEL_R, AXLE_L, MAX_WHEEL_W, V_MAX)

# Choose the pose source used for navigation.
# "odometry" uses wheel encoders and can drift.
# "ground_truth" uses the Supervisor's true pose.
# Comparing both helps evaluate the effect of position accuracy.
POSE_SOURCE = os.environ.get("NAV_POSE_SOURCE", "odometry")
OUT_DIR = os.path.join(DATA_DIR, f"run_{POSE_SOURCE}")
os.makedirs(OUT_DIR, exist_ok=True)

GOAL = planner.GOAL_WORLD
ROBOT_R = planner.ROBOT_RADIUS
WP_RADIUS = 0.06
GOAL_RADIUS = 0.05
MAX_SIM_TIME = 150.0
MAX_REPLANS = 8
REPLAN_COOLDOWN = 1.5

# IR sensor angles relative to the robot's heading.
IR_ANGLES = [-0.26, -0.79, -1.57, -2.62, 2.62, 1.57, 0.79, 0.26]
FRONT_IDX = [6, 7, 0, 1]
STOP_IDX = [7, 0]
IR_HIT = 115.0
IR_ESTOP = 600.0

# Approximate relationship between obstacle distance and IR reading.
IR_TABLE = [(0.0, 4095.0), (0.005, 2133.33), (0.01, 1465.73), (0.015, 601.46),
            (0.02, 383.84), (0.03, 234.93), (0.04, 158.03), (0.05, 120.0),
            (0.06, 104.09), (0.07, 67.19)]

# Adjust the wheelbase used for odometry to account for wheel slip.
ODOM_AXLE = float(os.environ.get("NAV_ODOM_AXLE", "0.0565"))

# Stop forward movement when turning sharply.
PIVOT_ERR = 0.6
PIVOT_W_MAX = 2.5
REPLAN_BACKUP_TIME = 0.6

# Camera settings for detecting coloured obstacles.
ROI_ROW0 = 56
SAT_MIN, VAL_MIN = 140, 35
HUE_BANDS = [(0, 10), (170, 179),   # Red
             (11, 35),              # Yellow
             (36, 85),              # Green
             (95, 135),             # Blue
             (136, 169)]            # Magenta
MIN_BLOB_AREA = 15
CONFIRM_AREA = 150
CONFIRM_BEARING = 0.45
EST_HALF_SIZE = 0.10
KNOWN_TOL = 0.04
ESTOP_RECOVER_AFTER = 3.0
BACKUP_TIME = 1.0


def detect_obstacles(img_bgr, fov):
    """Detect coloured objects and estimate their bearing."""
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    f = (w / 2.0) / math.tan(fov / 2.0)
    blobs, union = [], np.zeros((h, w), np.uint8)
    for h0, h1 in HUE_BANDS:
        mask = cv2.inRange(hsv, (h0, SAT_MIN, VAL_MIN), (h1, 255, 255))
        mask[:ROI_ROW0, :] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        union |= mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < MIN_BLOB_AREA:
                continue
            m = cv2.moments(c)
            if m["m00"] == 0:
                continue
            cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            bearing = -math.atan((cx - w / 2.0) / f)
            blobs.append(dict(cx=cx, cy=cy, area=area, bearing=bearing))
    return blobs, union


def save_debug_images(tag, img_bgr, mask, blobs):
    over = img_bgr.copy()
    for b in blobs:
        cv2.circle(over, (int(b["cx"]), int(b["cy"])), 4, (0, 255, 0), 1)
        cv2.putText(over, f"{math.degrees(b['bearing']):.0f}d", (int(b["cx"]) + 5, int(b["cy"])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    cv2.line(over, (0, ROI_ROW0), (over.shape[1], ROI_ROW0), (255, 255, 0), 1)
    up = lambda im: cv2.resize(im, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(OUT_DIR, f"perception_{tag}_raw.png"), up(img_bgr))
    cv2.imwrite(os.path.join(OUT_DIR, f"perception_{tag}_mask.png"), up(mask))
    cv2.imwrite(os.path.join(OUT_DIR, f"perception_{tag}_detections.png"), up(over))


def ir_to_distance(value):
    """Convert an IR sensor reading into an estimated distance."""
    if value <= IR_TABLE[-1][1] + 5.0:
        return None
    ds = [d for d, _ in IR_TABLE]
    vs = [v for _, v in IR_TABLE]
    return float(np.interp(value, vs[::-1], ds[::-1]))


def is_known(xy):
    """Check whether a detected obstacle is already in the map."""
    for (cx, cy, hx, hy) in planner.OBSTACLES:
        dx = max(abs(xy[0] - cx) - hx, 0.0)
        dy = max(abs(xy[1] - cy) - hy, 0.0)
        if math.hypot(dx, dy) <= KNOWN_TOL:
            return True
    return False


def fuse(ps_vals, blobs, pose):
    """Combine IR distance readings and camera bearings."""
    ir_front = max(ps_vals[i] for i in FRONT_IDX)
    ir_stop = max(ps_vals[i] for i in STOP_IDX)
    ir_hit = ir_front > IR_HIT
    estop = ir_stop > IR_ESTOP
    cand = [b for b in blobs if abs(b["bearing"]) < CONFIRM_BEARING and b["area"] >= CONFIRM_AREA]
    confirmed = ir_hit and bool(cand)

    obstacle_xy, ang, unknown = None, None, False
    if ir_hit:
        i_best = max(FRONT_IDX, key=lambda i: ps_vals[i])
        d = ir_to_distance(ps_vals[i_best])
        d = 0.04 if d is None else d

        # Use the IR sensor to select a direction and the camera to refine it.
        ir_ang = IR_ANGLES[i_best]
        if cand:
            pool = [b for b in cand if abs(b["bearing"] - ir_ang) < 0.6] or cand
            bearing = min(pool, key=lambda b: abs(b["bearing"] - ir_ang))["bearing"]
        else:
            bearing = ir_ang
        ang = pose[2] + bearing
        r = ROBOT_R + d
        obstacle_xy = (pose[0] + r * math.cos(ang), pose[1] + r * math.sin(ang))
        unknown = not is_known(obstacle_xy)
    return dict(ir_front=ir_front, ir_hit=ir_hit, estop=estop, confirmed=confirmed,
                obstacle_xy=obstacle_xy, ang=ang, unknown=unknown)


def add_obstacle(surface_xy, ang, robot_xy):
    """Add a detected obstacle and rebuild the occupancy grid."""
    cx = surface_xy[0] + EST_HALF_SIZE * math.cos(ang)
    cy = surface_xy[1] + EST_HALF_SIZE * math.sin(ang)
    planner.OBSTACLES.append((cx, cy, EST_HALF_SIZE, EST_HALF_SIZE))
    grid = planner.build_occupancy_grid()
    for gx in range(planner.GRID_N):
        for gy in range(planner.GRID_N):
            wx, wy = planner.grid_to_world(gx, gy)
            if math.hypot(wx - robot_xy[0], wy - robot_xy[1]) < 0.07:
                grid[gx, gy] = 0
    return grid


def make_plan(grid, start_xy, goal_xy):
    s = planner.world_to_grid(*start_xy)
    g = planner.world_to_grid(*goal_xy)
    path, nodes, rt = planner.astar(grid, s, g)
    if path is None:
        return None
    wps = [planner.grid_to_world(*c) for c in simplify_path(grid, path)]
    wps[0], wps[-1] = tuple(start_xy), tuple(goal_xy)
    return dict(waypoints=[list(map(float, w)) for w in wps], grid_path_len=len(path),
                nodes_explored=nodes, runtime_s=rt)


def main():
    robot = Supervisor()
    ts = int(robot.getBasicTimeStep())
    dt = ts / 1000.0

    ps = [robot.getDevice(f"ps{i}") for i in range(8)]
    for s in ps:
        s.enable(ts)
    camera = robot.getDevice("camera")
    camera.enable(2 * ts)
    cam_w, cam_h, fov = camera.getWidth(), camera.getHeight(), camera.getFov()
    lm, rm = robot.getDevice("left wheel motor"), robot.getDevice("right wheel motor")
    for m in (lm, rm):
        m.setPosition(float("inf"))
        m.setVelocity(0.0)
    lenc, renc = robot.getDevice("left wheel sensor"), robot.getDevice("right wheel sensor")
    lenc.enable(ts)
    renc.enable(ts)

    me = robot.getSelf()
    if me is None:
        print("ERROR: Supervisor node not available. Set 'supervisor TRUE' on the E-puck in the world file.")
        return

    def true_pose():
        p, R = me.getPosition(), me.getOrientation()
        return p[0], p[1], math.atan2(R[3], R[0])

    robot.step(ts)
    tx, ty, tth = true_pose()
    est = [tx, ty, tth]
    l_prev, r_prev = lenc.getValue(), renc.getValue()

    grid = planner.build_occupancy_grid()
    np.save(os.path.join(OUT_DIR, "nav_grid_initial.npy"), grid)
    plans = []
    plan = make_plan(grid, est[:2], GOAL)
    if plan is None:
        print("ERROR: no initial path found")
        return
    plan.update(t=0.0, reason="initial")
    plans.append(plan)
    wps = plan["waypoints"]
    print(f"[nav] pose source = {POSE_SOURCE}; initial plan: {len(wps)} waypoints, "
          f"{plan['nodes_explored']} A* nodes")

    pid = PID(*TUNED_GAINS)
    wp_idx, replans, last_replan = 1, 0, -99.0
    estop_since, saved_frame0, saved_event = None, False, False
    blobs, mask, img_bgr = [], None, None
    log, reached, step, state = [], False, 0, "TRACK"
    backup_until = -1.0

    try:
        while robot.step(ts) != -1:
            t = robot.getTime()
            step += 1

            # Update the estimated position using wheel encoders.
            l, r = lenc.getValue(), renc.getValue()
            dl, dr = (l - l_prev) * WHEEL_R, (r - r_prev) * WHEEL_R
            l_prev, r_prev = l, r
            d, dth = (dl + dr) / 2.0, (dr - dl) / ODOM_AXLE
            est[0] += d * math.cos(est[2] + dth / 2.0)
            est[1] += d * math.sin(est[2] + dth / 2.0)
            est[2] = wrap(est[2] + dth)
            tx, ty, tth = true_pose()
            loc = est if POSE_SOURCE == "odometry" else [tx, ty, tth]

            # Process camera images and detect coloured obstacles.
            if step % 2 == 0 or img_bgr is None:
                raw = camera.getImage()
                if raw:
                    img = np.frombuffer(raw, np.uint8).reshape((cam_h, cam_w, 4))
                    img_bgr = np.ascontiguousarray(img[:, :, :3])
                    blobs, mask = detect_obstacles(img_bgr, fov)
                    if not saved_frame0 and t > 0.3:
                        save_debug_images("frame0", img_bgr, mask, blobs)
                        saved_frame0 = True
            ps_vals = [s.getValue() for s in ps]
            fus = fuse(ps_vals, blobs, loc)

            # Replan when a new obstacle is detected.
            if fus["estop"]:
                estop_since = t if estop_since is None else estop_since
            else:
                estop_since = None
            estop_timeout = estop_since is not None and (t - estop_since) > 1.0
            if (estop_since is not None and (t - estop_since) > ESTOP_RECOVER_AFTER
                    and not fus["unknown"]):
                backup_until, estop_since = t + BACKUP_TIME, None
            if ((fus["confirmed"] or estop_timeout) and fus["unknown"]
                    and t - last_replan > REPLAN_COOLDOWN and replans < MAX_REPLANS):
                if not saved_event and img_bgr is not None:
                    save_debug_images("event_first_detection", img_bgr, mask, blobs)
                    saved_event = True
                grid = add_obstacle(fus["obstacle_xy"], fus["ang"], loc[:2])
                new_plan = make_plan(grid, loc[:2], GOAL)
                last_replan = t
                if new_plan is None:
                    print(f"[nav] t={t:.1f}s REPLAN FAILED (no path)")
                    break
                new_plan.update(t=float(t), reason="replan")
                plans.append(new_plan)
                wps, wp_idx, replans = new_plan["waypoints"], 1, replans + 1
                pid.reset()
                backup_until = t + REPLAN_BACKUP_TIME
                print(f"[nav] t={t:.1f}s unmapped obstacle at "
                      f"({fus['obstacle_xy'][0]:.2f},{fus['obstacle_xy'][1]:.2f}) -> replan #{replans}")

            # Use PID control to steer toward the current waypoint.
            while True:
                gx, gy = wps[wp_idx]
                dx, dy = gx - loc[0], gy - loc[1]
                dist = math.hypot(dx, dy)
                last = wp_idx == len(wps) - 1
                if dist < (GOAL_RADIUS if last else WP_RADIUS):
                    if last:
                        reached = True
                        break
                    wp_idx += 1
                    continue
                break
            if reached:
                break

            err = wrap(math.atan2(dy, dx) - loc[2])
            w_cmd = pid(err, dt)
            v_cmd = V_MAX * max(0.0, math.cos(err)) * min(1.0, dist / 0.15 + 0.3)
            if abs(err) > PIVOT_ERR:
                v_cmd = 0.0
                w_cmd = max(-PIVOT_W_MAX, min(PIVOT_W_MAX, w_cmd))
            state = "TRACK"
            if fus["ir_hit"]:
                v_cmd, state = v_cmd * 0.5, "SLOW"
            if fus["estop"]:
                v_cmd, state = 0.0, "ESTOP"
            if t < backup_until:
                v_cmd, w_cmd, state = -0.5 * V_MAX, 0.0, "BACKUP"

            wl = (v_cmd - w_cmd * AXLE_L / 2.0) / WHEEL_R
            wr = (v_cmd + w_cmd * AXLE_L / 2.0) / WHEEL_R
            limit = MAX_WHEEL_W * 0.999
            m = max(abs(wl), abs(wr))
            if m > limit:
                wl, wr = wl * limit / m, wr * limit / m
            lm.setVelocity(wl)
            rm.setVelocity(wr)

            log.append([round(t, 4), tx, ty, tth, est[0], est[1], est[2], len(plans) - 1, wp_idx,
                        err, v_cmd, w_cmd, wl, wr, fus["ir_front"], len(blobs),
                        int(fus["confirmed"]), int(fus["unknown"]), state])
            if step % 300 == 0:
                print(f"[nav] t={t:5.1f}s pos=({tx:.2f},{ty:.2f}) wp {wp_idx}/{len(wps) - 1} "
                      f"dist={dist:.2f} state={state}")
            if t > MAX_SIM_TIME:
                print("[nav] timeout - stopping")
                break
    finally:
        lm.setVelocity(0.0)
        rm.setVelocity(0.0)
        t_end = robot.getTime()
        tx, ty, tth = true_pose()
        header = ["t", "true_x", "true_y", "true_th", "est_x", "est_y", "est_th", "plan_id", "wp_idx",
                  "heading_err", "v_cmd", "w_cmd", "wl", "wr", "ir_front", "n_blobs",
                  "confirmed", "unknown_obstacle", "state"]
        with open(os.path.join(OUT_DIR, "nav_log.csv"), "w", newline="") as f:
            wr_ = csv.writer(f)
            wr_.writerow(header)
            wr_.writerows(log)
        with open(os.path.join(OUT_DIR, "nav_plans.json"), "w") as f:
            json.dump(plans, f, indent=2)
        np.save(os.path.join(OUT_DIR, "nav_grid_final.npy"), grid)
        with open(os.path.join(OUT_DIR, "nav_map_obstacles.json"), "w") as f:
            json.dump([list(o) for o in planner.OBSTACLES], f)
        summary = dict(reached=bool(reached), pose_source=POSE_SOURCE, t_end=float(t_end),
                       replans=int(replans),
                       final_true_xy=[float(tx), float(ty)],
                       final_goal_error_m=float(math.hypot(tx - GOAL[0], ty - GOAL[1])),
                       n_log_rows=len(log))
        with open(os.path.join(OUT_DIR, "nav_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[nav] finished: reached={reached}, replans={replans}, t={t_end:.1f}s, "
              f"goal error={summary['final_goal_error_m']:.3f} m. Logs in {OUT_DIR}")
        try:
            robot.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
        except Exception:
            pass


main()
