"""
swarm_agent.py - Controller for the four e-puck robots.

Each robot runs this controller to communicate with the others,
navigate the arena and visit its assigned sectors.

Modes:
    distributed - Robots share information and coordinate their plans.
    centralised - A leader creates and shares the plan.
    collect     - Robots collect sensor readings for training.
"""

import csv
import json
import math
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import swarm_core as sc  # noqa: E402


WHEEL_R, AXLE = 0.0205, 0.052
MAX_WHEEL = 6.0
N = sc.N_AGENTS
ROUND_T = 0.10
DISCOVER_ROUNDS = 12
PSO_ITERS = 100
PSO_PARTICLES_TOTAL = 40
CONS_PERIOD = 0.20
CALIB_STEPS = 16
COMPASS_DEFAULT = 3
COMPASS_TIE = 0.05
COLLECT_TIME = 120.0
DATA_DIR = os.path.join(HERE, "..", "..", "data")


def rx_text(rx):
    """Read a message from the receiver."""
    for fn in ("getString", "getBytes", "getData"):
        if hasattr(rx, fn):
            try:
                data = getattr(rx, fn)()
                if isinstance(data, (bytes, bytearray)):
                    data = data.decode("utf-8", "ignore")
                return data
            except Exception:
                continue
    return ""


def run(mode=None, robot=None):
    from_webots = robot is None

    if from_webots:
        from controller import Robot
        robot = Robot()

    mode = mode or (sys.argv[1] if len(sys.argv) > 1 else "distributed")
    ts = int(robot.getBasicTimeStep())
    name = robot.getName()
    ID = int("".join(ch for ch in name if ch.isdigit()) or 0)

    LEADER = N - 1
    NBRS = sc.neighbours(sc.ring_adjacency(N), ID)
    W_CONS = 1.0 / (1 + 2)

    # Set up the robot's motors and sensors.
    left = robot.getDevice("left wheel motor")
    right = robot.getDevice("right wheel motor")

    for motor in (left, right):
        motor.setPosition(float("inf"))
        motor.setVelocity(0.0)

    ps = [robot.getDevice(f"ps{i}") for i in range(8)]
    for sensor in ps:
        sensor.enable(ts)

    gps = robot.getDevice("gps")
    compass = robot.getDevice("compass")
    gps.enable(ts)
    compass.enable(ts)

    tx = robot.getDevice("swarm_tx")
    rx = robot.getDevice("swarm_rx")
    rx.enable(ts)

    # Load the trained neural network, or use the rule-based controller.
    wpath = os.path.join(HERE, "nn_weights.json")

    if os.path.exists(wpath):
        with open(wpath) as fh:
            policy = sc.MLP.from_dict(json.load(fh)).policy
        policy_name = "neural network"
    else:
        policy = sc.expert_policy
        policy_name = "rule set (nn_weights.json missing)"

    arbiter = sc.Arbiter(policy)

    print(
        f"[agent {ID}] mode={mode} policy={policy_name} "
        f"neighbours={NBRS}"
    )

    # Create files for recording the robot's activity.
    run_dir = os.path.join(DATA_DIR, f"swarm_run_{mode}")
    os.makedirs(run_dir, exist_ok=True)

    log_fh = open(
        os.path.join(run_dir, f"agent_{ID}.csv"),
        "w",
        newline=""
    )
    log = csv.writer(log_fh)
    log.writerow([
        "t", "state", "x", "y", "th", "target",
        "completed", "cons_total_est", "action", "ir_max"
    ])

    pso_fh = open(
        os.path.join(run_dir, f"pso_{ID}.csv"),
        "w",
        newline=""
    )
    pso_log = csv.writer(pso_fh)
    pso_log.writerow([
        "round", "t", "best_cost",
        "adopted_from_neighbour", "msgs_sent"
    ])

    collect_fh = None

    if mode == "collect":
        os.makedirs(DATA_DIR, exist_ok=True)
        collect_fh = open(
            os.path.join(DATA_DIR, f"ir_collect_{ID}.csv"),
            "w",
            newline=""
        )
        collect_w = csv.writer(collect_fh)
        collect_w.writerow(
            [f"ps{i}" for i in range(8)] + ["label"]
        )

    # Convert the requested speeds into wheel speeds.
    def drive(v, w):
        wl = (v - w * AXLE / 2.0) / WHEEL_R
        wr = (v + w * AXLE / 2.0) / WHEEL_R

        max_speed = max(abs(wl), abs(wr))

        if max_speed > MAX_WHEEL:
            wl = wl * MAX_WHEEL / max_speed
            wr = wr * MAX_WHEEL / max_speed

        left.setVelocity(wl)
        right.setVelocity(wr)

    # Send a message to the other robots.
    def send(msg):
        msg["id"] = ID
        tx.send(json.dumps(msg))

    # Read all waiting messages.
    def recv_all():
        messages = []

        while rx.getQueueLength() > 0:
            txt = rx_text(rx)
            rx.nextPacket()

            try:
                messages.append(json.loads(txt))
            except ValueError:
                pass

        return messages

    # Try different formulas to find the correct compass heading.
    heading_fns = [
        lambda c: math.atan2(-c[1], c[0]),
        lambda c: math.atan2(c[1], c[0]),
        lambda c: math.atan2(-c[0], c[1]),
        lambda c: math.atan2(c[0], c[1]),
    ]

    conv = [0]

    def pose():
        gx, gy, _ = gps.getValues()
        heading = heading_fns[conv[0]](compass.getValues())
        return gx, gy, heading

    # Initialise the variables used during the mission.
    state = "CALIB"
    calib_i = 0
    calib_p0 = None

    known = {}
    leader_best = ID
    round_i = 0
    next_round = 0.0

    swarm = None
    plan = None
    plan_t0 = None
    plan_t1 = None

    msgs_sent = 0
    order = []
    ptr = 0
    dwell_until = None
    completed = 0

    centres = sc.sector_centres()

    x_est = 0.0
    last_c = 0
    nbr_x = {}
    next_cons = 0.0
    cons_total = 0.0

    step_i = 0
    rng_local = random.Random(100 + ID)
    wander_target = None
    wander_until = 0.0
    done_printed = False
    last_action = 0

    while robot.step(ts) != -1:
        t = robot.getTime()
        gx, gy, _ = gps.getValues()

        if math.isnan(gx):
            continue

        step_i += 1
        raw = [sensor.getValue() for sensor in ps]
        feats = sc.ir_features(raw)

        # Check the compass while moving a short distance.
        if state == "CALIB":
            if calib_p0 is None:
                calib_p0 = (gx, gy)

            drive(0.05, 0.0)
            calib_i += 1

            if calib_i >= CALIB_STEPS:
                drive(0.0, 0.0)

                gh = math.atan2(
                    gy - calib_p0[1],
                    gx - calib_p0[0]
                )

                disp = math.hypot(
                    gy - calib_p0[1],
                    gx - calib_p0[0]
                )

                c = compass.getValues()

                errs = [
                    abs(sc.wrap_angle(fn(c) - gh))
                    for fn in heading_fns
                ]

                ranked = [int(i) for i in np.argsort(errs)]
                best, second = ranked[0], ranked[1]

                if errs[second] - errs[best] < COMPASS_TIE:
                    if COMPASS_DEFAULT in (best, second):
                        conv[0] = COMPASS_DEFAULT
                    else:
                        conv[0] = best
                else:
                    conv[0] = best

                best_error = errs[conv[0]]

                state = "WANDER" if mode == "collect" else "ALLOC"
                next_round = t + ROUND_T

            continue

        x, y, th = pose()

        # Move around the arena and collect sensor readings.
        if state == "WANDER":
            if t >= COLLECT_TIME + 1.0:
                drive(0, 0)

                if collect_fh:
                    collect_fh.close()
                    collect_fh = None
                    print(
                        f"[agent {ID}] IR data written to "
                        f"data/ir_collect_{ID}.csv"
                    )

                continue

            if (
                    wander_target is None
                    or t > wander_until
                    or math.hypot(
                wander_target[0] - x,
                wander_target[1] - y
            ) < 0.1
            ):
                wander_target = (
                    rng_local.uniform(-0.8, 0.8),
                    rng_local.uniform(-0.8, 0.8)
                )
                wander_until = t + 10.0

            v, w = arbiter.step(
                feats, (x, y, th), wander_target
            )
            drive(v, w)

            if collect_fh and step_i % 3 == 0:
                collect_w.writerow(
                    [f"{r:.1f}" for r in raw]
                    + [int(sc.expert_policy(feats))]
                )

            continue

        # Exchange robot positions and agree on the leader.
        if state == "DISCOVER":
            known[ID] = (x, y)

            for msg in recv_all():
                sender = msg.get("id")

                if msg.get("t") == "disc":
                    if sender in NBRS:
                        leader_best = max(
                            leader_best, msg["lead"]
                        )

                        for key, value in msg.get("pos", {}).items():
                            known[int(key)] = tuple(value)

            if t >= next_round:
                pos_payload = {
                    str(key): list(value)
                    for key, value in known.items()
                }

                send({
                    "t": "disc",
                    "lead": leader_best,
                    "pos": pos_payload
                })

                msgs_sent += 1
                round_i += 1
                next_round += ROUND_T

                if round_i >= DISCOVER_ROUNDS:
                    missing = [
                        i for i in range(N)
                        if i not in known
                    ]

                    if missing:
                        round_i = DISCOVER_ROUNDS - 1
                        drive(0, 0)
                        continue

                    LEADER = leader_best
                    state = "ALLOC"
                    round_i = 0
                    next_round = t + ROUND_T
                    plan_t0 = t

            drive(0, 0)
            continue

        # Assign the four yellow sectors to the four robots.
        elif state == "ALLOC":
            drive(0, 0)

            plan = [-1] * len(centres)
            plan[10] = 0
            plan[11] = 1
            plan[14] = 2
            plan[15] = 3

            mine = [
                sector
                for sector in range(len(plan))
                if plan[sector] == ID
            ]

            if len(mine) > 0 and hasattr(sc, "nn_tour"):
                order = sc.nn_tour((x, y), mine, centres)[0]
            else:
                order = list(mine)

            ptr = 0
            completed = 0
            print(f"[agent {ID}] mode={mode} -> upper-right yellow allocation; sectors {order}")

            state = "EXEC"
            next_cons = t
            continue

        # Exchange progress estimates and visit assigned sectors.
        for msg in recv_all():
            if (
                    msg.get("t") == "cons"
                    and msg["id"] in NBRS
            ):
                nbr_x[msg["id"]] = msg["x"]

        if t >= next_cons:
            x_est += (
                    W_CONS
                    * sum(
                nbr_x[j] - x_est
                for j in NBRS
                if j in nbr_x
            )
                    + (completed - last_c)
            )

            last_c = completed
            send({"t": "cons", "x": x_est})
            msgs_sent += 1
            next_cons += CONS_PERIOD
            cons_total = N * x_est

        action_name = "-"

        if state == "EXEC":
            if len(order) == 0 or ptr >= len(order):
                drive(0, 0)

                if cons_total >= len(centres) - 0.6:
                    state = "DONE"

            elif dwell_until is not None:
                drive(0, 0)

                if t >= dwell_until:
                    dwell_until = None
                    ptr += 1
                    completed += 1

            else:
                s_cur = order[ptr]
                target = centres[s_cur]

                # Calculate movement towards the current target.
                v, w = arbiter.step(
                    feats, (x, y, th), target
                )

                # Turn away from nearby obstacles.
                if max(raw) > 120:
                    v = 0.0
                    left_side = sum(raw[0:4])
                    right_side = sum(raw[4:8])

                    if left_side > right_side:
                        w = 1.5
                    else:
                        w = -1.5

                elif max(raw) > 70:
                    v = 0.02
                    left_side = sum(raw[0:4])
                    right_side = sum(raw[4:8])

                    if left_side > right_side:
                        w = 0.8
                    else:
                        w = -0.8

                last_action = arbiter.last_action
                drive(v, w)

                # Start scanning when the robot reaches the sector centre.
                if math.hypot(
                        target[0] - x,
                        target[1] - y
                ) < sc.VISIT_RADIUS:
                    dwell_until = t + sc.SECTOR_WORK[s_cur]

        if state == "DONE":
            drive(0, 0)

            if not done_printed:
                done_printed = True
                print(
                    f"[agent {ID}] t={t:.1f}s all sectors "
                    f"scanned (consensus estimate "
                    f"{cons_total:.2f}/4)"
                )

        if step_i % 5 == 0:
            tgt_s = (
                order[ptr]
                if state == "EXEC" and ptr < len(order)
                else -1
            )

            log.writerow([
                f"{t:.3f}",
                state,
                f"{x:.4f}",
                f"{y:.4f}",
                f"{th:.4f}",
                tgt_s,
                completed,
                f"{cons_total:.3f}",
                action_name,
                f"{max(raw):.0f}"
            ])

            if step_i % 100 == 0:
                log_fh.flush()
                pso_fh.flush()

    log_fh.close()
    pso_fh.close()


if __name__ == "__main__":
    run()