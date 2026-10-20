
"""
Component A - Step 4: Integrated Autonomous Navigation

This Webots controller makes the e-puck detect obstacles, plan a path,
and move toward the goal.

Required files:
- Analysis/Planning.py
- Analysis/control_system.py

The controller uses three main parts:

Perception:
The camera detects coloured obstacles and estimates their direction.
The IR sensors estimate distance and detect nearby obstacles.
Both sensors are combined to improve obstacle detection.

Planning:
A* finds a path to the goal.
If a new obstacle is detected, the map is updated and a new path is planned.

Control:
PID control helps the robot turn toward each waypoint.
Wheel encoders estimate the robot's position.

The controller saves navigation logs, plans, maps, and camera images.
"""

import os
import sys
import csv
import json
import math

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import cv2
from controller import Supervisor

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(PROJECT, "Analysis"))

DATA_DIR = os.path.join(PROJECT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

import Planning as planner
from control_system import (
    PID, simplify_path, wrap, TUNED_GAINS,
    WHEEL_R, AXLE_L, MAX_WHEEL_W, V_MAX
)

# Choose how the robot estimates its position.
# "odometry" uses wheel sensors and may drift.
# "ground_truth" uses the exact position from Webots.
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

# Angles of the eight IR sensors.
IR_ANGLES = [-0.26, -0.79, -1.57, -2.62, 2.62, 1.57, 0.79, 0.26]

FRONT_IDX = [6, 7, 0, 1]
STOP_IDX = [7, 0]

IR_HIT = 115.0
IR_ESTOP = 600.0

# Approximate IR readings for different distances.
IR_TABLE = [
    (0.0, 4095.0),
    (0.005, 2133.33),
    (0.01, 1465.73),
    (0.015, 601.46),
    (0.02, 383.84),
    (0.03, 234.93),
    (0.04, 158.03),
    (0.05, 120.0),
    (0.06, 104.09),
    (0.07, 67.19)
]

# Camera detection settings.
ROI_ROW0 = 56
SAT_MIN, VAL_MIN = 110, 35
MIN_BLOB_AREA = 15
CONFIRM_AREA = 150
CONFIRM_BEARING = 0.45

EST_HALF_SIZE = 0.10
KNOWN_TOL = 0.04
ESTOP_RECOVER_AFTER = 3.0
BACKUP_TIME = 1.0


def detect_obstacles(img_bgr, fov):
    """Detect coloured objects and calculate their direction."""

    h, w = img_bgr.shape[:2]

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(
        hsv,
        (0, SAT_MIN, VAL_MIN),
        (180, 255, 255)
    )

    mask[:ROI_ROW0, :] = 0

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8)
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    f = (w / 2.0) / math.tan(fov / 2.0)
    blobs = []

    for c in contours:
        area = cv2.contourArea(c)

        if area < MIN_BLOB_AREA:
            continue

        m = cv2.moments(c)

        if m["m00"] == 0:
            continue

        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]

        # Calculate the object's direction from the centre of the image.
        bearing = -math.atan((cx - w / 2.0) / f)

        blobs.append(
            dict(cx=cx, cy=cy, area=area, bearing=bearing)
        )

    return blobs, mask


def save_debug_images(tag, img_bgr, mask, blobs):
    """Save images showing what the camera detects."""

    over = img_bgr.copy()

    for b in blobs:
        cv2.circle(
            over,
            (int(b["cx"]), int(b["cy"])),
            4,
            (0, 255, 0),
            1
        )

        cv2.putText(
            over,
            f"{math.degrees(b['bearing']):.0f}d",
            (int(b["cx"]) + 5, int(b["cy"])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (0, 255, 0),
            1
        )

    # Mark the part of the image used for detection.
    cv2.line(
        over,
        (0, ROI_ROW0),
        (over.shape[1], ROI_ROW0),
        (255, 255, 0),
        1
    )

    # Enlarge images to make the results easier to see.
    up = lambda im: cv2.resize(
        im,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_NEAREST
    )

    cv2.imwrite(
        os.path.join(OUT_DIR, f"perception_{tag}_raw.png"),
        up(img_bgr)
    )

    cv2.imwrite(
        os.path.join(OUT_DIR, f"perception_{tag}_mask.png"),
        up(mask)
    )

    cv2.imwrite(
        os.path.join(OUT_DIR, f"perception_{tag}_detections.png"),
        up(over)
    )


def ir_to_distance(value):
    """Convert an IR sensor reading into an estimated distance."""

    if value <= IR_TABLE[-1][1] + 5.0:
        return None

    ds = [d for d, _ in IR_TABLE]
    vs = [v for _, v in IR_TABLE]

    return float(np.interp(value, vs[::-1], ds[::-1]))


def is_known(xy):
    """Check whether an obstacle is already on the map."""

    for (cx, cy, hx, hy) in planner.OBSTACLES:
        dx = max(abs(xy[0] - cx) - hx, 0.0)
        dy = max(abs(xy[1] - cy) - hy, 0.0)

        if math.hypot(dx, dy) <= KNOWN_TOL:
            return True

    return False


def fuse(ps_vals, blobs, pose):
    """Combine sensor readings to estimate an obstacle's position."""

    ir_front = max(ps_vals[i] for i in FRONT_IDX)
    ir_stop = max(ps_vals[i] for i in STOP_IDX)

    # Check whether an obstacle is close or requires an emergency stop.
    ir_hit = ir_front > IR_HIT
    estop = ir_stop > IR_ESTOP

    # Keep camera detections that are large and roughly in front.
    cand = [
        b for b in blobs
        if abs(b["bearing"]) < CONFIRM_BEARING
        and b["area"] >= CONFIRM_AREA
    ]

    confirmed = ir_hit and bool(cand)

    obstacle_xy, ang, unknown = None, None, False

    if ir_hit:
        # Use the strongest front sensor to estimate distance.
        i_best = max(FRONT_IDX, key=lambda i: ps_vals[i])

        d = ir_to_distance(ps_vals[i_best])
        d = 0.04 if d is None else d

        # Use the camera direction when available.
        bearing = (
            max(cand, key=lambda b: b["area"])["bearing"]
            if cand else IR_ANGLES[i_best]
        )

        ang = pose[2] + bearing
        r = ROBOT_R + d

        obstacle_xy = (
            pose[0] + r * math.cos(ang),
            pose[1] + r * math.sin(ang)
        )

        unknown = not is_known(obstacle_xy)

    return dict(
        ir_front=ir_front,
        ir_hit=ir_hit,
        estop=estop,
        confirmed=confirmed,
        obstacle_xy=obstacle_xy,
        ang=ang,
        unknown=unknown
    )


def add_obstacle(surface_xy, ang, robot_xy):
    """Add a new obstacle to the map and rebuild the grid."""

    # Estimate the obstacle's centre from its detected surface.
    cx = surface_xy[0] + EST_HALF_SIZE * math.cos(ang)
    cy = surface_xy[1] + EST_HALF_SIZE * math.sin(ang)

    planner.OBSTACLES.append(
        (cx, cy, EST_HALF_SIZE, EST_HALF_SIZE)
    )

    grid = planner.build_occupancy_grid()

    # Keep the robot's current cell clear.
    for gx in range(planner.GRID_N):
        for gy in range(planner.GRID_N):
            wx, wy = planner.grid_to_world(gx, gy)

            if math.hypot(wx - robot_xy[0], wy - robot_xy[1]) < 0.07:
                grid[gx, gy] = 0

    return grid


def make_plan(grid, start_xy, goal_xy):
    """Use A* to find a path from the start to the goal."""

    s = planner.world_to_grid(*start_xy)
    g = planner.world_to_grid(*goal_xy)

    path, nodes, rt = planner.astar(grid, s, g)

    if path is None:
        return None

    wps = [
        planner.grid_to_world(*c)
        for c in simplify_path(grid, path)
    ]

    # Use the exact start and goal positions.
    wps[0], wps[-1] = tuple(start_xy), tuple(goal_xy)

    return dict(
        waypoints=[list(map(float, w)) for w in wps],
        grid_path_len=len(path),
        nodes_explored=nodes,
        runtime_s=rt
    )


def main():
    """Set up the robot and run the navigation loop."""

    robot = Supervisor()

    ts = int(robot.getBasicTimeStep())
    dt = ts / 1000.0

    # Enable the IR sensors.
    ps = [robot.getDevice(f"ps{i}") for i in range(8)]

    for s in ps:
        s.enable(ts)

    # Enable the camera.
    camera = robot.getDevice("camera")
    camera.enable(2 * ts)

    cam_w = camera.getWidth()
    cam_h = camera.getHeight()
    fov = camera.getFov()

    # Set up the wheel motors.
    lm = robot.getDevice("left wheel motor")
    rm = robot.getDevice("right wheel motor")

    for m in (lm, rm):
        m.setPosition(float("inf"))
        m.setVelocity(0.0)

    # Enable the wheel sensors.
    lenc = robot.getDevice("left wheel sensor")
    renc = robot.getDevice("right wheel sensor")

    lenc.enable(ts)
    renc.enable(ts)

    me = robot.getSelf()

    if me is None:
        print(
            "ERROR: Supervisor node not available. "
            "Set 'supervisor TRUE' on the E-puck in the world file."
        )
        return

    def true_pose():
        """Get the robot's actual position from Webots."""

        p = me.getPosition()
        R = me.getOrientation()

        return p[0], p[1], math.atan2(R[3], R[0])

    # Wait for the sensors to return readings.
    robot.step(ts)

    tx, ty, tth = true_pose()

    # Start with the known position, then use wheel odometry.
    est = [tx, ty, tth]

    l_prev = lenc.getValue()
    r_prev = renc.getValue()

    # Build the starting map and find the first path.
    grid = planner.build_occupancy_grid()

    np.save(
        os.path.join(OUT_DIR, "nav_grid_initial.npy"),
        grid
    )

    plans = []
    plan = make_plan(grid, est[:2], GOAL)

    if plan is None:
        print("ERROR: no initial path found")
        return

    plan.update(t=0.0, reason="initial")
    plans.append(plan)

    wps = plan["waypoints"]

    print(
        f"[nav] pose source = {POSE_SOURCE}; "
        f"initial plan: {len(wps)} waypoints, "
        f"{plan['nodes_explored']} A* nodes"
    )

    # Set up the PID controller and navigation variables.
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

            # Update the estimated position using the wheel sensors.
            l, r = lenc.getValue(), renc.getValue()

            dl = (l - l_prev) * WHEEL_R
            dr = (r - r_prev) * WHEEL_R

            l_prev, r_prev = l, r

            d = (dl + dr) / 2.0
            dth = (dr - dl) / AXLE_L

            est[0] += d * math.cos(est[2] + dth / 2.0)
            est[1] += d * math.sin(est[2] + dth / 2.0)
            est[2] = wrap(est[2] + dth)

            tx, ty, tth = true_pose()

            # Choose which position estimate to use.
            loc = est if POSE_SOURCE == "odometry" else [tx, ty, tth]

            # Read the camera image and detect obstacles.
            if step % 2 == 0 or img_bgr is None:
                raw = camera.getImage()

                if raw:
                    img = np.frombuffer(
                        raw,
                        np.uint8
                    ).reshape((cam_h, cam_w, 4))

                    img_bgr = np.ascontiguousarray(img[:, :, :3])

                    blobs, mask = detect_obstacles(img_bgr, fov)

                    if not saved_frame0 and t > 0.3:
                        save_debug_images(
                            "frame0",
                            img_bgr,
                            mask,
                            blobs
                        )
                        saved_frame0 = True

            # Read the IR sensors and combine their readings with the camera.
            ps_vals = [s.getValue() for s in ps]
            fus = fuse(ps_vals, blobs, loc)

            # Check whether the robot needs to stop or find a new path.
            if fus["estop"]:
                estop_since = t if estop_since is None else estop_since
            else:
                estop_since = None

            estop_timeout = (
                estop_since is not None
                and (t - estop_since) > 1.0
            )

            # Back up if the emergency stop lasts too long.
            if (
                estop_since is not None
                and (t - estop_since) > ESTOP_RECOVER_AFTER
                and not fus["unknown"]
            ):
                backup_until = t + BACKUP_TIME
                estop_since = None

            # Add a new obstacle to the map and calculate a new path.
            if (
                (fus["confirmed"] or estop_timeout)
                and fus["unknown"]
                and t - last_replan > REPLAN_COOLDOWN
                and replans < MAX_REPLANS
            ):
                if not saved_event and img_bgr is not None:
                    save_debug_images(
                        "event_first_detection",
                        img_bgr,
                        mask,
                        blobs
                    )
                    saved_event = True

                grid = add_obstacle(
                    fus["obstacle_xy"],
                    fus["ang"],
                    loc[:2]
                )

                new_plan = make_plan(grid, loc[:2], GOAL)
                last_replan = t

                if new_plan is None:
                    print(f"[nav] t={t:.1f}s REPLAN FAILED (no path)")
                    break

                new_plan.update(t=float(t), reason="replan")
                plans.append(new_plan)

                wps = new_plan["waypoints"]
                wp_idx = 1
                replans += 1

                # Reset PID after changing the path.
                pid.reset()

                print(
                    f"[nav] t={t:.1f}s unmapped obstacle at "
                    f"({fus['obstacle_xy'][0]:.2f},"
                    f"{fus['obstacle_xy'][1]:.2f}) "
                    f"-> replan #{replans}"
                )

            # Move to the next waypoint when the current one is reached.
            while True:
                gx, gy = wps[wp_idx]

                dx = gx - loc[0]
                dy = gy - loc[1]
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

            # Use PID to turn toward the current waypoint.
            err = wrap(math.atan2(dy, dx) - loc[2])
            w_cmd = pid(err, dt)

            # Reduce speed when the robot needs to turn sharply or is close.
            v_cmd = (
                V_MAX
                * max(0.0, math.cos(err))
                * min(1.0, dist / 0.15 + 0.3)
            )

            state = "TRACK"

            # Slow down when an obstacle is detected nearby.
            if fus["ir_hit"]:
                v_cmd, state = v_cmd * 0.5, "SLOW"

            # Stop if an obstacle is too close.
            if fus["estop"]:
                v_cmd, state = 0.0, "ESTOP"

            # Reverse briefly when backing away from an obstacle.
            if t < backup_until:
                v_cmd, w_cmd, state = -0.5 * V_MAX, 0.0, "BACKUP"

            # Convert movement commands into wheel speeds.
            wl = (v_cmd - w_cmd * AXLE_L / 2.0) / WHEEL_R
            wr = (v_cmd + w_cmd * AXLE_L / 2.0) / WHEEL_R

            # Keep wheel speeds within the allowed limit.
            m = max(abs(wl), abs(wr))

            if m > MAX_WHEEL_W:
                wl = wl * MAX_WHEEL_W / m
                wr = wr * MAX_WHEEL_W / m

            lm.setVelocity(wl)
            rm.setVelocity(wr)

            # Save the current navigation data.
            log.append([
                round(t, 4),
                tx, ty, tth,
                est[0], est[1], est[2],
                len(plans) - 1,
                wp_idx,
                err, v_cmd, w_cmd,
                wl, wr,
                fus["ir_front"],
                len(blobs),
                int(fus["confirmed"]),
                int(fus["unknown"]),
                state
            ])

            if step % 300 == 0:
                print(
                    f"[nav] t={t:5.1f}s "
                    f"pos=({tx:.2f},{ty:.2f}) "
                    f"wp {wp_idx}/{len(wps) - 1} "
                    f"dist={dist:.2f} state={state}"
                )

            if t > MAX_SIM_TIME:
                print("[nav] timeout - stopping")
                break

    finally:
        # Stop the motors when navigation ends.
        lm.setVelocity(0.0)
        rm.setVelocity(0.0)

        t_end = robot.getTime()
        tx, ty, tth = true_pose()

        # Save the navigation log.
        header = [
            "t", "true_x", "true_y", "true_th",
            "est_x", "est_y", "est_th",
            "plan_id", "wp_idx", "heading_err",
            "v_cmd", "w_cmd", "wl", "wr",
            "ir_front", "n_blobs",
            "confirmed", "unknown_obstacle", "state"
        ]

        with open(
            os.path.join(OUT_DIR, "nav_log.csv"),
            "w",
            newline=""
        ) as f:
            wr_ = csv.writer(f)
            wr_.writerow(header)
            wr_.writerows(log)

        # Save the paths and final map.
        with open(
            os.path.join(OUT_DIR, "nav_plans.json"),
            "w"
        ) as f:
            json.dump(plans, f, indent=2)

        np.save(
            os.path.join(OUT_DIR, "nav_grid_final.npy"),
            grid
        )

        with open(
            os.path.join(OUT_DIR, "nav_map_obstacles.json"),
            "w"
        ) as f:
            json.dump(
                [list(o) for o in planner.OBSTACLES],
                f
            )

        # Save a summary of the run.
        summary = dict(
            reached=bool(reached),
            pose_source=POSE_SOURCE,
            t_end=float(t_end),
            replans=int(replans),
            final_true_xy=[float(tx), float(ty)],
            final_goal_error_m=float(
                math.hypot(tx - GOAL[0], ty - GOAL[1])
            ),
            n_log_rows=len(log)
        )

        with open(
            os.path.join(OUT_DIR, "nav_summary.json"),
            "w"
        ) as f:
            json.dump(summary, f, indent=2)

        print(
            f"[nav] finished: reached={reached}, "
            f"replans={replans}, t={t_end:.1f}s, "
            f"goal error={summary['final_goal_error_m']:.3f} m. "
            f"Logs in {OUT_DIR}"
        )

        try:
            robot.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
        except Exception:
            pass


main()
