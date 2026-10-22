"""
mock_webots_test.py - Tests the real swarm_agent.py and
swarm_supervisor.py controllers using a small fake Webots environment.

This helps find logic and API errors without installing Webots.
It does not replace testing in the actual Webots simulator.

Run:
    python mock_webots_test.py
    python mock_webots_test.py distributed --quick
    python mock_webots_test.py centralised
"""

import json
import math
import os
import sys
import threading
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "controllers", "swarm_agent"))
sys.path.insert(0, os.path.join(HERE, "..", "controllers", "swarm_supervisor"))

import swarm_core as sc  # noqa: E402
from swarm_sim import SwarmSim  # noqa: E402

MODE = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "distributed"
TS = 32
N = sc.N_AGENTS


class World:
    def __init__(self):
        self.poses = [list(s) for s in sc.STARTS]
        self.vel = [(0.0, 0.0)] * N
        self.t = 0.0
        self.inbox = [[] for _ in range(N)]
        self.sim = SwarmSim(sc.STARTS, sc.sector_centres(), None, np.random.default_rng(0))
        self.bar = threading.Barrier(N + 2)  # Robots, supervisor and driver synchronise each step.
        self.stop = False
        self.pending = []


W = World()


class Dev:
    def __init__(self, fn=None):
        self.fn = fn
        self.v = 0.0

    def enable(self, *_): pass
    def setPosition(self, *_): pass
    def setVelocity(self, v): self.v = v
    def getValue(self): return self.fn()


class Receiver(Dev):
    def __init__(self, i): self.i = i
    def enable(self, *_): pass
    def getQueueLength(self): return len(W.inbox[self.i])
    def getString(self): return W.inbox[self.i][0]
    def nextPacket(self): W.inbox[self.i].pop(0)


class Emitter(Dev):
    def __init__(self, i): self.i = i
    def send(self, s):
        W.pending.append((self.i, s))


class MockRobot:
    SIMULATION_MODE_PAUSE = 0

    def __init__(self, i):
        self.i = i
        self.devs = {
            "left wheel motor": Dev(), "right wheel motor": Dev(),
            "gps": Dev(lambda: None), "compass": Dev(lambda: None),
            "swarm_tx": Emitter(i), "swarm_rx": Receiver(i)}
        for k in range(8):
            self.devs[f"ps{k}"] = Dev(lambda k=k: float(self._ir()[k]))
        self.devs["gps"].getValues = lambda: [W.poses[i][0], W.poses[i][1], 0.0]
        # Return the robot's heading in the compass format used by this test.
        self.devs["compass"].getValues = lambda: [math.cos(W.poses[i][2]), math.sin(W.poses[i][2]), 0.0]
        self._cache = (-1, None)

    def _ir(self):
        step = round(W.t / (TS / 1000))
        if self._cache[0] != step:
            f = W.sim.sense(self.i, [tuple(p) for p in W.poses])
            self._cache = (step, sc.IR_BASELINE + f * sc.IR_SCALE)
        return self._cache[1]

    def getBasicTimeStep(self): return TS
    def getName(self): return f"epuck{self.i}"
    def getDevice(self, n): return self.devs[n]
    def getTime(self): return W.t

    def step(self, ts):
        W.bar.wait()  # Wait for all controllers to issue their commands.
        W.bar.wait()  # Wait for physics and message delivery to finish.
        return -1 if W.stop else 0


class MockSupervisor(MockRobot):
    def __init__(self):
        self.i = -1
        self.nodes = []

    def getFromDef(self, name):
        i = int(name[-1])
        return types.SimpleNamespace(getPosition=lambda: [W.poses[i][0], W.poses[i][1], 0.0])

    def simulationSetMode(self, m): W.stop = True


def physics_step():
    dt = TS / 1000
    for i in range(N):
        wl, wr = W.vel[i]
        v = (wl + wr) / 2 * 0.0205
        w = (wr - wl) * 0.0205 / 0.052
        th = W.poses[i][2] + w * dt
        cand = (W.poses[i][0] + v * math.cos(th) * dt, W.poses[i][1] + v * math.sin(th) * dt)
        others = [(W.poses[j][0], W.poses[j][1]) for j in range(N) if j != i]
        if W.sim._blocked(cand, others) is None:
            W.poses[i][0], W.poses[i][1] = cand
        W.poses[i][2] = sc.wrap_angle(th)
    W.t += dt

    # Deliver messages to every robot except the sender.
    # The mock uses unrestricted radio range but keeps the controller's topology.
    for (src, s) in W.pending:
        for j in range(N):
            if j != src:
                W.inbox[j].append(s)
    W.pending.clear()


def main():
    import swarm_agent
    import swarm_supervisor
    robots = [MockRobot(i) for i in range(N)]
    sup = MockSupervisor()
    threads = [threading.Thread(target=swarm_agent.run, args=(MODE, r), daemon=True) for r in robots]
    threads.append(threading.Thread(target=swarm_supervisor.run, args=(MODE, sup), daemon=True))
    for th in threads:
        th.start()
    max_steps = int(300 / (TS / 1000))
    for k in range(max_steps):
        W.bar.wait()  # Wait for the controllers to finish their commands.
        for i, r in enumerate(robots):
            W.vel[i] = (r.devs["left wheel motor"].v, r.devs["right wheel motor"].v)
        physics_step()
        if W.stop:
            W.bar.wait()
            break
        W.bar.wait()
    W.stop = True
    for th in threads:
        th.join(timeout=2)


if __name__ == "__main__":
    # Keep mock test output separate from real Webots results.
    import swarm_agent, swarm_supervisor
    tmp = os.path.join(HERE, "..", "data", "_mock")
    swarm_agent.DATA_DIR = tmp
    swarm_supervisor.DATA_DIR = tmp
    main()
