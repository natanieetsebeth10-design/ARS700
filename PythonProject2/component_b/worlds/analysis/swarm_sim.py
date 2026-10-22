"""
swarm_sim.py - Lightweight model of the Webots swarm environment.

Used for Monte Carlo experiments with many random seeds and failure levels.
It models the arena, obstacles, e-puck IR sensors and robot control logic.

Webots remains the main simulator. This file is used to collect statistics
more quickly during offline experiments.
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Go up two levels: worlds/analysis -> component_b
COMPONENT_B = os.path.dirname(os.path.dirname(HERE))

SWARM_AGENT_DIR = os.path.join(
    COMPONENT_B, "controllers", "swarm_agent"
)

if SWARM_AGENT_DIR not in sys.path:
    sys.path.insert(0, SWARM_AGENT_DIR)

import swarm_core as sc

_RAY_D = np.linspace(0.0, sc.IR_RANGE, 29)


class SwarmSim:
    def __init__(self, starts, centres, policy, rng, obstacles=None, dt=0.032,
                 noise_sigma=15.0, noise_rel=0.03):
        self.starts = [tuple(s) for s in starts]
        self.centres = centres
        self.obstacles = sc.OBSTACLES if obstacles is None else obstacles
        self.policy, self.rng, self.dt = policy, rng, dt
        self.noise_sigma, self.noise_rel = noise_sigma, noise_rel
        self.half = sc.ARENA / 2.0

    def sense(self, i, poses):
        x, y, th = poses[i]
        ang = th + sc.IR_ANGLES
        dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1)
        origin = np.array([x, y]) + 0.034 * dirs
        P = origin[:, None, :] + _RAY_D[None, :, None] * dirs[:, None, :]
        hit = np.zeros(P.shape[:2], bool)
        for (ox, oy, h) in self.obstacles:
            hit |= (np.abs(P[..., 0] - ox) < h) & (np.abs(P[..., 1] - oy) < h)
        hit |= (np.abs(P[..., 0]) > self.half) | (np.abs(P[..., 1]) > self.half)
        for j, pj in enumerate(poses):
            if j != i:
                hit |= np.hypot(P[..., 0] - pj[0], P[..., 1] - pj[1]) < sc.ROBOT_RADIUS
        first = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
        dist = np.where(first >= 0, _RAY_D[np.maximum(first, 0)], 1.0)
        raw = sc.ir_reading_from_distance(dist)
        raw = raw * (1 + self.rng.normal(0, self.noise_rel, 8)) + self.rng.normal(0, self.noise_sigma, 8)
        return sc.ir_features(np.clip(raw, 0, 4095))

    def _blocked(self, p, others):
        r = sc.ROBOT_RADIUS
        if abs(p[0]) > self.half - r or abs(p[1]) > self.half - r:
            return "wall"
        for (ox, oy, h) in self.obstacles:
            dx = max(abs(p[0] - ox) - h, 0.0)
            dy = max(abs(p[1] - oy) - h, 0.0)
            if math.hypot(dx, dy) < r:
                return "obstacle"
        for q in others:
            if math.hypot(p[0] - q[0], p[1] - q[1]) < 2 * r:
                return "robot"
        return None

    def run(self, sets, t_max=160.0):
        """
        Each agent receives a list of sector indices to visit and scan.
        Robots remain still while scanning each sector.
        Returns success, coverage, time, collisions and robot paths.
        """
        n = len(self.starts)
        poses = [list(s) for s in self.starts]
        orders = [sc.nn_tour(self.starts[i][:2], sets[i], self.centres)[0] for i in range(n)]
        ptr = [0] * n
        dwell = [0] * n
        arbs = [sc.Arbiter(self.policy) for _ in range(n)]
        done = set()
        in_contact = [None] * n
        coll = dict(wall=0, obstacle=0, robot=0)
        paths = [[(p[0], p[1])] for p in poses]
        t, steps = 0.0, int(t_max / self.dt)
        all_sectors = set(range(len(self.centres)))
        for k in range(steps):
            for i in range(n):
                if ptr[i] >= len(orders[i]):
                    continue
                s_cur = orders[i][ptr[i]]
                if dwell[i] > 0:
                    dwell[i] -= 1
                    if dwell[i] == 0:
                        done.add(s_cur)
                        ptr[i] += 1
                    continue
                tgt = self.centres[s_cur]
                f = self.sense(i, [tuple(p) for p in poses])
                v, w = arbs[i].step(f, tuple(poses[i]), tgt)
                th = poses[i][2] + w * self.dt
                cand = (poses[i][0] + v * math.cos(th) * self.dt,
                        poses[i][1] + v * math.sin(th) * self.dt)
                others = [(poses[j][0], poses[j][1]) for j in range(n) if j != i]
                kind = self._blocked(cand, others)
                if kind is None:
                    poses[i][0], poses[i][1] = cand
                    in_contact[i] = None
                else:
                    if in_contact[i] != kind:
                        coll[kind] += 1
                    in_contact[i] = kind
                poses[i][2] = sc.wrap_angle(th)
                if math.hypot(poses[i][0] - tgt[0], poses[i][1] - tgt[1]) < sc.VISIT_RADIUS:
                    dwell[i] = max(1, int(sc.SECTOR_WORK[s_cur] / self.dt))
            if k % 5 == 0:
                for i in range(n):
                    paths[i].append((poses[i][0], poses[i][1]))
            t = (k + 1) * self.dt
            if done >= all_sectors or all(ptr[i] >= len(orders[i]) for i in range(n)):
                break
        return dict(success=done >= all_sectors, coverage=len(done) / len(all_sectors),
                    time=t, collisions=sum(coll.values()), coll=coll, paths=paths)


def single_robot_trial(policy, rng, t_max=60.0, noise_sigma=15.0):
    """Test whether one robot reaches a random goal sector without collisions."""
    centres = sc.sector_centres()
    a, b = rng.choice(len(centres), 2, replace=False)
    s = centres[a]
    th0 = rng.uniform(-math.pi, math.pi)
    sim = SwarmSim([(s[0], s[1], th0)], centres, policy, rng, noise_sigma=noise_sigma)
    out = sim.run([[int(b)]], t_max=t_max)
    reached = out["coverage"] > 0
    return dict(success=bool(reached and out["collisions"] == 0), reached=bool(reached),
                collisions=out["collisions"], time=out["time"])