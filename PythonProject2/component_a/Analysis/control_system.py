"""
Component A - Step 3: Control System Design

This script simulates the e-puck robot and controls its movement.

What the code does:
1. Simulates how the robot moves and turns.
2. Uses a PID controller to control the robot's heading.
3. Tests different PID settings and compares the results.
4. Uses A* to find a path and makes the robot follow it.
5. Saves graphs and results for the report.

"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

import Planning as planner


# Set the robot's wheel sizes and movement limits.
WHEEL_R = 0.0205
AXLE_L = 0.052
MAX_WHEEL_W = 6.28
V_MAX = WHEEL_R * MAX_WHEEL_W
W_MAX = 2 * V_MAX / AXLE_L

# Set the motor delay, turning bias and simulation time step.
TAU = 0.08
TURN_BIAS = 0.15
DT = 0.016


def wrap(a):
    # Keep the angle between -pi and pi.
    return (a + np.pi) % (2 * np.pi) - np.pi


class DiffDriveRobot:
    # Simulate the robot's movement, motors and turning.

    def __init__(self, x=0.0, y=0.0, theta=0.0, bias=TURN_BIAS):
        self.x, self.y, self.theta = x, y, theta
        self.v, self.w = 0.0, 0.0
        self.bias = bias

    def step(self, v_cmd, w_cmd, dt=DT):
        # Keep the speed and turning commands within their limits.
        v_cmd = np.clip(v_cmd, 0.0, V_MAX)
        w_cmd = np.clip(w_cmd, -W_MAX, W_MAX)

        # Simulate the delay in the motors.
        a = dt / (TAU + dt)
        self.v += a * (v_cmd - self.v)
        self.w += a * (w_cmd - self.w)

        # Update the robot's position and direction.
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
        self.theta = wrap(self.theta + (self.w + self.bias) * dt)

    def wheel_speeds(self):
        # Calculate the left and right wheel speeds.
        wl = (self.v - self.w * AXLE_L / 2) / WHEEL_R
        wr = (self.v + self.w * AXLE_L / 2) / WHEEL_R
        return wl, wr


# Set the angle limit for the PID integral term.
I_ZONE = np.deg2rad(15.0)


class PID:
    def __init__(self, kp, ki, kd, out_limit=W_MAX, i_limit=1.0, i_zone=I_ZONE):
        # Store the PID settings and limits.
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_limit, self.i_limit, self.i_zone = out_limit, i_limit, i_zone
        self.integral = 0.0
        self.prev_err = None

    def reset(self):
        # Reset the controller's stored values.
        self.integral, self.prev_err = 0.0, None

    def __call__(self, err, dt=DT):
        # Only add to the integral when the error is small.
        # This helps prevent the controller from overshooting.
        if abs(err) < self.i_zone:
            self.integral = np.clip(
                self.integral + err * dt,
                -self.i_limit,
                self.i_limit
            )
        else:
            self.integral = 0.0

        # Calculate how quickly the error is changing.
        d = 0.0 if self.prev_err is None else (err - self.prev_err) / dt
        self.prev_err = err

        # Combine the P, I and D terms to calculate the output.
        out = self.kp * err + self.ki * self.integral + self.kd * d

        # Keep the output within the allowed turning speed.
        return float(np.clip(out, -self.out_limit, self.out_limit))


# Set four different PID settings to compare their performance.
GAIN_SETS = {
    "A: P only, low gain (Kp=1.5)": (1.5, 0.0, 0.0),
    "B: P only, high gain (Kp=12)": (12.0, 0.0, 0.0),
    "C: PID tuned (Kp=8, Ki=6, Kd=0.5)": (8.0, 6.0, 0.5),
    "D: PID, excessive Ki (Kp=8, Ki=60, Kd=0.5)": (8.0, 60.0, 0.5),
}

# Set the target turning angle and test duration.
STEP_RAD = np.deg2rad(45.0)
T_SIM = 6.0


def heading_step_response(gains):
    # Create a robot and PID controller using the selected settings.
    robot = DiffDriveRobot(theta=0.0)
    pid = PID(*gains)

    # Prepare arrays to store the time and heading.
    n = int(T_SIM / DT)
    t = np.arange(n) * DT
    theta = np.zeros(n)

    for i in range(n):
        # Calculate the difference between the target and current heading.
        err = wrap(STEP_RAD - robot.theta)

        # Use the PID controller to calculate the turning command.
        w_cmd = pid(err)

        # Turn on the spot without moving forward.
        robot.step(0.0, w_cmd)
        theta[i] = robot.theta

    return t, theta


def step_metrics(t, y, ref):
    # Calculate rise time, overshoot, settling time and final error.
    y = np.asarray(y)

    # Find how long the response takes to rise from 10% to 90%.
    try:
        t10 = t[np.where(y >= 0.1 * ref)[0][0]]
        t90 = t[np.where(y >= 0.9 * ref)[0][0]]
        rise = t90 - t10
    except IndexError:
        rise = float("nan")

    # Calculate how far the heading goes beyond the target.
    overshoot = max(0.0, (y.max() - ref) / ref * 100.0)

    # Find when the heading stays within 2% of the target.
    outside = np.where(np.abs(y - ref) > 0.02 * ref)[0]

    if len(outside) == 0:
        settle = 0.0
    elif outside[-1] >= len(t) - 1:
        settle = float("nan")
    else:
        settle = t[outside[-1] + 1]

    # Calculate the remaining error using the last 0.5 seconds.
    n_last = int(0.5 / DT)
    sse = ref - float(np.mean(y[-n_last:]))

    return rise, overshoot, settle, sse


def fmt(v, unit=""):
    # Format numbers for display and show n/a if the value is unavailable.
    return "n/a" if (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}{unit}"


def run_tuning_study(out_dir):
    # Prepare the results and graph for comparing PID settings.
    results = {}
    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Draw the target heading and its acceptable 2% range.
    ax.axhline(np.rad2deg(STEP_RAD), color="k", ls="--", lw=1, label="Reference (45°)")
    band = 0.02 * np.rad2deg(STEP_RAD)
    ax.axhspan(
        np.rad2deg(STEP_RAD) - band,
        np.rad2deg(STEP_RAD) + band,
        color="grey",
        alpha=0.15,
        label="±2% band"
    )

    # Display the heading response results in a table.
    print("\n" + "=" * 92)
    print("HEADING STEP RESPONSE (45° step, includes 0.15 rad/s wheel-mismatch bias)")
    print("=" * 92)
    print(f"{'Gain set':<44}{'Rise(s)':>9}{'OS(%)':>9}{'Settle(s)':>11}{'SSE(deg)':>11}")

    rows = []

    for name, g in GAIN_SETS.items():
        # Test the current PID settings.
        t, th = heading_step_response(g)

        # Calculate the performance measurements.
        rise, os_, settle, sse = step_metrics(t, th, STEP_RAD)

        # Add the response to the graph.
        ax.plot(t, np.rad2deg(th), lw=2, label=name)

        # Print the results for this set of gains.
        print(
            f"{name:<44}{fmt(rise):>9}{os_:>9.2f}"
            f"{fmt(settle):>11}{np.rad2deg(sse):>11.3f}"
        )

        rows.append((name, g, rise, os_, settle, np.rad2deg(sse)))

        # Store the results so they can be saved later.
        results[name] = dict(
            kp=g[0],
            ki=g[1],
            kd=g[2],
            rise_s=rise,
            overshoot_pct=os_,
            settle_s=settle,
            sse_deg=float(np.rad2deg(sse))
        )

    # Explain the steady-state offset in the first test.
    print("Note: for set A the 'OS' figure is the steady-state offset caused by the")
    print("      wheel-mismatch bias (P-only cannot reject it), not an oscillation.")

    # Add labels and save the tuning graph.
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heading (deg)")
    ax.set_title("Effect of PID tuning on heading step response")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pid_tuning_response.png"), dpi=150)

    # Save the tuning measurements to a CSV file.
    with open(os.path.join(out_dir, "pid_tuning_metrics.csv"), "w") as f:
        f.write("gain_set,kp,ki,kd,rise_time_s,overshoot_pct,settling_time_s,sse_deg\n")

        for name, g, rise, os_, settle, sse in rows:
            f.write(
                f"\"{name}\",{g[0]},{g[1]},{g[2]},"
                f"{rise:.4f},{os_:.3f},{settle:.4f},{sse:.4f}\n"
            )

    return results


# Choose the tuned PID settings and waypoint distances.
TUNED_GAINS = GAIN_SETS["C: PID tuned (Kp=8, Ki=6, Kd=0.5)"]
WP_RADIUS = 0.06
GOAL_RADIUS = 0.05


def line_of_sight(grid, a, b):
    # Check whether a straight path between two cells is free of obstacles.
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2) + 1

    for k in range(n + 1):
        s = k / n
        gx = int(round(a[0] + s * (b[0] - a[0])))
        gy = int(round(a[1] + s * (b[1] - a[1])))

        # Stop if an obstacle blocks the straight path.
        if grid[gx, gy] == 1:
            return False

    return True


def simplify_path(grid, path):
    # Remove unnecessary grid points when a straight route is clear.
    wps = [path[0]]
    i = 0

    while i < len(path) - 1:
        j = len(path) - 1

        # Find the furthest point that can be reached in a straight line.
        while j > i + 1 and not line_of_sight(grid, path[i], path[j]):
            j -= 1

        wps.append(path[j])
        i = j

    return wps


def track_path(waypoints_xy, start_pose, gains, bias=TURN_BIAS, t_max=120.0):
    # Create the robot and controller for following the path.
    robot = DiffDriveRobot(*start_pose, bias=bias)
    pid = PID(*gains)

    # Start at the first waypoint after the robot's current position.
    wp_idx = 1
    log = []
    t = 0.0

    while t < t_max:
        # Find the direction and distance to the current waypoint.
        tx, ty = waypoints_xy[wp_idx]
        dx, dy = tx - robot.x, ty - robot.y
        dist = np.hypot(dx, dy)

        # Check whether the current waypoint is the final goal.
        last = wp_idx == len(waypoints_xy) - 1

        # Move to the next waypoint when the current one is close enough.
        if dist < (GOAL_RADIUS if last else WP_RADIUS):
            if last:
                break

            wp_idx += 1
            continue

        # Calculate the heading error and the required turning speed.
        err = wrap(np.arctan2(dy, dx) - robot.theta)
        w_cmd = pid(err)

        # Slow down when the robot is facing away from the waypoint.
        v_cmd = V_MAX * max(0.0, np.cos(err)) * min(1.0, dist / 0.15 + 0.3)

        # Move the robot and save its position and movement data.
        robot.step(v_cmd, w_cmd)
        log.append((t, robot.x, robot.y, robot.theta, err, robot.v, robot.w))
        t += DT

    return np.array(log), (t < t_max)


def cross_track_error(log_xy, waypoints_xy):
    # Measure how far the robot moves away from the planned path.
    pts = np.asarray(waypoints_xy)
    errs = []

    for x, y in log_xy:
        best = np.inf

        # Check the distance to each section of the planned path.
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            ab = b - a

            # Find the closest point on this section of the path.
            tt = np.clip(np.dot([x, y] - a, ab) / np.dot(ab, ab), 0, 1)
            proj = a + tt * ab

            best = min(best, np.hypot(x - proj[0], y - proj[1]))

        errs.append(best)

    return np.array(errs)


def min_clearance(log_xy):
    # Find the smallest gap between the robot and any real obstacle.
    # A negative result means the robot collided with an obstacle.
    best = np.inf

    for x, y in log_xy:
        for (ox, oy, hw, hh) in planner.OBSTACLES:
            dx = max(abs(x - ox) - hw, 0.0)
            dy = max(abs(y - oy) - hh, 0.0)

            best = min(best, np.hypot(dx, dy) - planner.ROBOT_RADIUS)

    return best


def run_path_tracking(out_dir):
    # Build the arena grid and find the A* path.
    grid = planner.build_occupancy_grid()
    start = planner.world_to_grid(*planner.START_WORLD)
    goal = planner.world_to_grid(*planner.GOAL_WORLD)

    path, nodes, rt = planner.astar(grid, start, goal)

    # Simplify the path to reduce the number of waypoints.
    wps_grid = simplify_path(grid, path)
    wps = [planner.grid_to_world(*c) for c in wps_grid]

    # Use the exact start and goal positions from the planning script.
    wps[0] = planner.START_WORLD
    wps[-1] = planner.GOAL_WORLD

    # Start with the robot facing the positive x direction.
    start_pose = (planner.START_WORLD[0], planner.START_WORLD[1], 0.0)

    # Follow the path using the tuned PID controller.
    log, reached = track_path(wps, start_pose, TUNED_GAINS)

    # Extract the recorded time and robot positions.
    t = log[:, 0]
    xy = log[:, 1:3]

    # Measure how closely the robot follows the planned path.
    cte = cross_track_error(xy, wps)

    # Calculate the actual distance travelled and planned path length.
    path_len = float(np.sum(np.hypot(*np.diff(xy, axis=0).T)))
    planned_len = float(
        sum(
            np.hypot(
                wps[i + 1][0] - wps[i][0],
                wps[i + 1][1] - wps[i][1]
            )
            for i in range(len(wps) - 1)
        )
    )

    # Display the path tracking results.
    print("\n" + "=" * 60)
    print("PATH TRACKING RESULT (tuned PID, A* path with shortcutting)")
    print("=" * 60)
    print(f"Waypoints after simplification : {len(wps)} (from {len(path)} grid cells)")
    print(f"Goal reached                   : {reached}")
    print(f"Time to goal (s)               : {t[-1]:.1f}")
    print(f"Planned polyline length (m)    : {planned_len:.3f}")
    print(f"Actual distance travelled (m)  : {path_len:.3f}")
    print(f"Mean cross-track error (m)     : {cte.mean():.4f}")
    print(f"Max cross-track error (m)      : {cte.max():.4f}")
    print(f"RMS cross-track error (m)      : {np.sqrt(np.mean(cte**2)):.4f}")

    # Check the minimum distance between the robot and obstacles.
    clearance = float(min_clearance(xy))
    print(
        f"Min clearance to obstacles (m) : {clearance:.4f}  "
        f"({'no collision' if clearance > 0 else 'COLLISION'})"
    )

    # Draw the planned path and the robot's actual movement.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]

    # Display the grid and obstacles in the background.
    ax.imshow(
        grid.T,
        origin="lower",
        cmap="Greys",
        alpha=0.5,
        extent=[-1.5, 1.5, -1.5, 1.5]
    )

    wx, wy = zip(*wps)

    # Show the planned path, actual path, start and goal.
    ax.plot(wx, wy, "b--o", lw=2, label="Planned (A*, simplified)")
    ax.plot(xy[:, 0], xy[:, 1], "r-", lw=2, label="Actual (PID tracked)")
    ax.plot(*planner.START_WORLD, "go", ms=10, label="Start")
    ax.plot(*planner.GOAL_WORLD, "b*", ms=15, label="Goal")

    ax.set_title("Planned vs actual trajectory")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8)

    # Draw a graph showing how the tracking error changes over time.
    ax = axes[1]
    ax.plot(t, cte, color="purple", lw=1.8, label="Cross-track error (m)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cross-track error (m)")

    # Show the heading error on a second vertical axis.
    ax2 = ax.twinx()
    ax2.plot(
        t,
        np.rad2deg(log[:, 4]),
        color="orange",
        lw=1.2,
        alpha=0.8,
        label="Heading error (deg)"
    )
    ax2.set_ylabel("Heading error (deg)")
    ax.set_title("Tracking error over time")

    # Combine the labels from both axes into one legend.
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # Adjust the graphs and save the image.
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "path_tracking.png"), dpi=150)

    # Return the measurements so they can be saved later.
    return dict(
        waypoints=wps,
        time_to_goal_s=float(t[-1]),
        planned_length_m=planned_len,
        actual_length_m=path_len,
        mean_cte_m=float(cte.mean()),
        max_cte_m=float(cte.max()),
        rms_cte_m=float(np.sqrt(np.mean(cte**2))),
        min_clearance_m=clearance
    )


if __name__ == "__main__":
    # Create the folder for saving the results.
    OUT = "../data"
    os.makedirs(OUT, exist_ok=True)

    # Run the PID tuning tests and path tracking simulation.
    tuning = run_tuning_study(OUT)
    tracking = run_path_tracking(OUT)

    # Save the controller settings and results to a JSON file.
    with open(os.path.join(OUT, "control_results.json"), "w") as f:
        json.dump(
            {
                "tuned_gains": dict(
                    kp=TUNED_GAINS[0],
                    ki=TUNED_GAINS[1],
                    kd=TUNED_GAINS[2]
                ),
                "tuning_metrics": tuning,
                "tracking": tracking
            },
            f,
            indent=2
        )

    print(
        "\nSaved to data/: pid_tuning_response.png, pid_tuning_metrics.csv,"
        " path_tracking.png, control_results.json"
    )

    plt.show()