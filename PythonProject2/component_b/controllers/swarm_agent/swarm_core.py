"""
swarm_core.py - Shared algorithms for Component B (multi-agent swarm).

Uses only Python and NumPy, so the same code can run in Webots and
in the offline Matplotlib experiments.

Sections:
1. Arena and task settings
2. Communication, consensus and leader election
3. Task allocation
4. Particle Swarm Optimisation
5. IR sensors and obstacle avoidance
6. Neural network
"""

import math
import numpy as np


# 1. Arena and task settings

ARENA = 2.0                      # Width and height of the arena in metres
GRID = 4                         # Number of sectors along each side
N_AGENTS = 4                     # Number of robots
ROBOT_RADIUS = 0.037             # Radius of an e-puck robot in metres
VISIT_RADIUS = 0.12              # Maximum distance for counting a sector as visited

# Obstacles are placed between sectors so their centres remain accessible.
# Each obstacle is defined by its x position, y position and half-size.
OBSTACLES = [(0.0, 0.0, 0.08), (-0.5, 0.5, 0.08), (0.5, -0.5, 0.08)]

# Starting positions and headings for the four robots.
STARTS = [(-0.85, -0.85, math.pi / 4),
          (0.85, -0.85, 3 * math.pi / 4),
          (0.85, 0.85, -3 * math.pi / 4),
          (-0.85, 0.85, -math.pi / 4)]


# Each sector has a scanning time. Sectors 10, 11, 14 and 15
# take longer to scan, so the workload needs to be shared between robots.
SECTOR_WORK = [1.5] * (GRID * GRID)
for _s in (10, 11, 14, 15):
    SECTOR_WORK[_s] = 10.0

V_TRAVEL = 0.07                  # Estimated robot travel speed in metres per second


def sector_centres(grid=GRID, size=ARENA):
    """Return the centre coordinates of all sectors."""
    c = (np.arange(grid) + 0.5) * size / grid - size / 2.0
    return [(float(x), float(y)) for y in c for x in c]


# 2. Communication, consensus and leader election

def ring_adjacency(n):
    """Create a network where each robot communicates with its two neighbours."""
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        A[i, (i + 1) % n] = 1
        A[i, (i - 1) % n] = 1
    return A


def full_adjacency(n):
    """Create a network where every robot can communicate with every other robot."""
    return np.ones((n, n), dtype=int) - np.eye(n, dtype=int)


def path_adjacency(n):
    """Create a communication network arranged in a line."""
    A = np.zeros((n, n), dtype=int)
    for i in range(n - 1):
        A[i, i + 1] = A[i + 1, i] = 1
    return A


def star_adjacency(n, hub=0):
    """Create a network where one robot acts as the central communication hub."""
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        if i != hub:
            A[hub, i] = A[i, hub] = 1
    return A


def neighbours(A, i):
    """Return the IDs of the robots connected to robot i."""
    return [int(j) for j in np.nonzero(A[i])[0]]


def metropolis_weights(A):
    """Calculate communication weights that preserve the average value."""
    n = len(A)
    deg = A.sum(axis=1)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and A[i, j]:
                W[i, j] = 1.0 / (1 + max(deg[i], deg[j]))
        W[i, i] = 1.0 - W[i].sum()
    return W


def consensus_rate(W):
    """Estimate how quickly the robots approach the same value."""
    ev = np.sort(np.abs(np.linalg.eigvals(W)))[::-1]
    return float(ev[1])


def average_consensus(x0, A, iters, link_drop=0.0, rng=None):
    """
    Let connected robots repeatedly update their values using their neighbours.

    link_drop is the probability that a communication link fails in a round.
    The weights are recalculated when links fail.
    Returns the values recorded at each iteration.
    """
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x0, float).copy()
    hist = [x.copy()]
    n = len(x)

    for _ in range(iters):
        At = A.copy()
        if link_drop > 0:
            for i in range(n):
                for j in range(i + 1, n):
                    if At[i, j] and rng.random() < link_drop:
                        At[i, j] = At[j, i] = 0

        x = metropolis_weights(At) @ x
        hist.append(x.copy())

    return np.array(hist)


def leader_election(A, ids=None, max_rounds=None):
    """
    Find the robot with the highest ID through neighbour communication.

    Each robot shares its highest known ID with its neighbours.
    Returns the ID history and the round when all robots agree.
    """
    n = len(A)
    ids = list(range(n)) if ids is None else list(ids)
    best = np.array(ids)
    hist = [best.copy()]
    max_rounds = max_rounds or n
    agree_at = 0 if len(set(best)) == 1 else None

    for r in range(1, max_rounds + 1):
        new = best.copy()
        for i in range(n):
            for j in neighbours(A, i):
                new[i] = max(new[i], best[j])

        best = new
        hist.append(best.copy())

        if agree_at is None and len(set(best)) == 1:
            agree_at = r

    return np.array(hist), agree_at


# 3. Task allocation

def nn_tour(start, idx, centres):
    """Find a visiting order by repeatedly choosing the nearest unvisited sector."""
    px, py = start[0], start[1]
    rem = list(idx)
    order, length = [], 0.0

    while rem:
        best_k, best_d = 0, 1e9

        for k, s in enumerate(rem):
            d = math.hypot(centres[s][0] - px, centres[s][1] - py)
            if d < best_d:
                best_k, best_d = k, d

        s = rem.pop(best_k)
        length += best_d
        px, py = centres[s]
        order.append(s)

    return order, length


def agent_times(assign, starts_xy, centres, n_agents):
    """Estimate how long each robot needs to travel and scan its assigned sectors."""
    times = []

    for a in range(n_agents):
        idx = [s for s in range(len(assign)) if assign[s] == a]
        L = nn_tour(starts_xy[a], idx, centres)[1]
        times.append(L / V_TRAVEL + sum(SECTOR_WORK[s] for s in idx))

    return times


def allocation_cost(assign, starts_xy, centres, n_agents):
    """Calculate the overall mission time, with a small penalty for total effort."""
    T = agent_times(assign, starts_xy, centres, n_agents)
    return max(T) + 0.02 * sum(T)


def make_cost_fn(starts_xy, centres, n_agents):
    """Create a function that evaluates a sector assignment."""
    starts_xy = [tuple(s[:2]) for s in starts_xy]
    return lambda assign: allocation_cost(assign, starts_xy, centres, n_agents)


# 4. Particle Swarm Optimisation

class PSOSwarm:
    """
    Optimise which robot should visit each sector.

    Each particle stores an assignment of sectors to robots.
    During each update, it can keep its current assignment, copy its
    personal best, copy the swarm's best assignment or choose randomly.

    The inertia weight decreases over time to reduce exploration later.
    adopt() allows a robot to accept a better solution from a neighbour.
    """

    def __init__(self, n_agents, n_sectors, cost_fn, n_particles, rng,
                 total_iters=60, w_start=0.5, w_end=0.2, c1=0.4, c2=0.2, mutation=0.1):
        self.na, self.D, self.cost_fn, self.P, self.rng = n_agents, n_sectors, cost_fn, n_particles, rng
        self.T, self.w0, self.w1 = max(total_iters, 1), w_start, w_end
        self.c1, self.c2, self.mut = c1, c2, mutation

        self.x = rng.integers(0, n_agents, (n_particles, n_sectors))
        self.pbest = self.x.copy()
        self.pfit = np.array([cost_fn(p) for p in self.x])

        g = int(np.argmin(self.pfit))
        self.gbest, self.gfit = self.pbest[g].copy(), float(self.pfit[g])
        self.evals = n_particles
        self.t = 0

    @property
    def best_assign(self):
        """Return the best sector assignment found so far."""
        return self.gbest.copy()

    def step(self):
        """Update the particles and keep track of the best solution."""
        w = self.w0 + (self.w1 - self.w0) * min(self.t / self.T, 1.0)
        probs = np.array([w, self.c1, self.c2, self.mut])
        cum = np.cumsum(probs / probs.sum())

        u = self.rng.random((self.P, self.D))
        rand_agent = self.rng.integers(0, self.na, (self.P, self.D))

        new = np.where(u < cum[0], self.x,
              np.where(u < cum[1], self.pbest,
              np.where(u < cum[2], self.gbest[None, :], rand_agent)))

        self.x = new

        for p in range(self.P):
            f = self.cost_fn(self.x[p])

            if f < self.pfit[p]:
                self.pfit[p], self.pbest[p] = f, self.x[p].copy()

                if f < self.gfit:
                    self.gfit, self.gbest = float(f), self.x[p].copy()

        self.evals += self.P
        self.t += 1

        return self.gfit

    def adopt(self, assign, fit):
        """Accept a neighbour's assignment if it has a lower cost."""
        if fit < self.gfit - 1e-12:
            self.gbest = np.asarray(assign, int).copy()
            self.gfit = float(fit)
            return True

        return False


def run_centralised_pso(starts, centres, n_agents, total_particles, iters, rng, loss=0.0):
    """
    Run PSO on one leader robot.

    The leader estimates the starting positions, optimises the allocation
    and sends the result to the other robots. Lost messages may cause the
    leader to use an estimated position or a robot to miss its assignment.
    """
    n = n_agents
    leader = n - 1
    believed = []

    for a in range(n):
        if a == leader or rng.random() >= loss:
            believed.append(tuple(starts[a][:2]))
        else:
            believed.append((0.0, 0.0))

    swarm = PSOSwarm(n, len(centres), make_cost_fn(believed, centres, n), total_particles,
                     rng, total_iters=iters)

    hist = [swarm.gfit]

    for _ in range(iters):
        hist.append(swarm.step())

    received = [a == leader or rng.random() >= loss for a in range(n)]

    return dict(assign=swarm.best_assign.copy(), hist=np.array(hist), received=received,
                msgs=2 * (n - 1), evals=swarm.evals, leader=leader)


def run_distributed_pso(starts, centres, n_agents, particles_per_agent, iters, A, rng, loss=0.0):
    """
    Let every robot run its own PSO swarm.

    Robots exchange their best assignments with their neighbours each round.
    A robot adopts a neighbour's solution when it has a lower cost.
    Returns the best assignments, fitness history and communication totals.
    """
    n = n_agents
    cost = make_cost_fn(starts, centres, n)

    swarms = [PSOSwarm(n, len(centres), cost, particles_per_agent,
                       np.random.default_rng(int(rng.integers(1 << 31))), total_iters=iters)
              for _ in range(n)]

    hist = [[s.gfit for s in swarms]]
    msgs = 0

    for _ in range(iters):
        for s in swarms:
            s.step()

        snap = [(s.best_assign.copy(), s.gfit) for s in swarms]

        for i in range(n):
            for j in neighbours(A, i):
                msgs += 1

                if rng.random() >= loss:
                    swarms[i].adopt(*snap[j])

        hist.append([s.gfit for s in swarms])

    return dict(assigns=[s.best_assign.copy() for s in swarms], hist=np.array(hist),
                msgs=msgs, evals=sum(s.evals for s in swarms))


def executed_sets(result, n_agents):
    """Return the sectors each robot is expected to visit."""
    if "assigns" in result:
        return [[s for s in range(len(result["assigns"][i])) if result["assigns"][i][s] == i]
                for i in range(n_agents)]

    a = result["assign"]

    return [[s for s in range(len(a)) if a[s] == i] if result["received"][i] else []
            for i in range(n_agents)]


def mission_metrics(sets, starts, centres, n_sectors=None):
    """Calculate the planned coverage and estimated finishing time."""
    n_sectors = n_sectors or len(centres)
    covered = set(s for lst in sets for s in lst)

    times = [nn_tour(starts[i][:2], lst, centres)[1] / V_TRAVEL + sum(SECTOR_WORK[s] for s in lst)
             for i, lst in enumerate(sets)]

    return dict(coverage=len(covered) / n_sectors, makespan=max(times), times=times)


def greedy_assignment(starts, centres, n_agents):
    """Assign each sector to the robot whose starting position is closest."""
    a = []

    for c in centres:
        d = [math.hypot(c[0] - s[0], c[1] - s[1]) for s in starts]
        a.append(int(np.argmin(d)))

    return np.array(a)


# 5. IR sensors and obstacle avoidance

# Angles of the eight e-puck proximity sensors relative to the robot's heading.
# Positive angles point to the left.
IR_ANGLES = np.array([-0.30, -0.80, -1.57, -2.64, 2.64, 1.57, 0.80, 0.30])
IR_RANGE = 0.07                   # Approximate sensor range in metres

# Approximate relationship between distance and the raw sensor reading.
_LUT_D = np.array([0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
_LUT_V = np.array([4095, 2133, 1465, 601, 383, 234, 158, 120, 104, 70], float)

IR_BASELINE = 70.0
IR_SCALE = 1500.0

ACTIONS = ["forward", "left", "right", "reverse"]
ACT_FWD, ACT_LEFT, ACT_RIGHT, ACT_REV = 0, 1, 2, 3


def ir_reading_from_distance(d):
    """Estimate a proximity sensor reading from the distance to an obstacle."""
    return np.interp(d, _LUT_D, _LUT_V)


def ir_features(raw):
    """Convert raw sensor readings into values between zero and one."""
    return np.clip((np.asarray(raw, float) - IR_BASELINE) / IR_SCALE, 0.0, 1.0)


def expert_policy(f):
    """
    Choose a movement action from the sensor readings.

    The robot moves forward when the path is clear, reverses when an
    obstacle is very close in front, and otherwise turns away from danger.
    """
    f = np.asarray(f, float)

    L = f[..., 7] + 0.8 * f[..., 6] + 0.3 * f[..., 5]
    R = f[..., 0] + 0.8 * f[..., 1] + 0.3 * f[..., 2]

    danger = np.maximum(L, R)
    front = np.maximum(f[..., 0], f[..., 7])

    act = np.where(danger < 0.12, ACT_FWD,
                   np.where(front > 0.85, ACT_REV,
                            np.where(L > R, ACT_RIGHT, ACT_LEFT)))

    return act if act.ndim else int(act)


def wrap_angle(a):
    """Keep an angle within the range -pi to pi."""
    return (a + math.pi) % (2 * math.pi) - math.pi


class Arbiter:
    """
    Combine movement towards a target with obstacle avoidance.

    The avoidance policy can be a neural network or the expert controller.
    Turning and reversing take priority over moving towards the target.
    After avoiding an obstacle, the robot moves forward briefly before
    trying to follow the target again.

    step() returns the linear and angular speeds.
    """

    def __init__(self, policy, v_max=0.10, w_turn=2.0, w_max=2.2, kp=2.5,
                 turn_hold=5, straight_hold=26, rev_hold=5):
        self.policy = policy
        self.v_max, self.w_turn, self.w_max, self.kp = v_max, w_turn, w_max, kp

        self.TURN_HOLD, self.STRAIGHT_HOLD, self.REV_HOLD = turn_hold, straight_hold, rev_hold
        self.turn_hold = self.straight_hold = self.rev_hold = 0
        self.turn_dir = 1
        self.last_action = ACT_FWD

    def step(self, feats, pose, target):
        """Calculate the next movement command using sensor readings and the target."""
        a = int(self.policy(feats))
        self.last_action = a

        if self.rev_hold > 0:
            self.rev_hold -= 1

            if self.rev_hold == 0:
                self.turn_hold = self.TURN_HOLD * 2

            return -0.05, 0.0

        if a == ACT_REV:
            self.rev_hold = self.REV_HOLD
            self.turn_dir = -self.turn_dir
            return -0.05, 0.0

        if a in (ACT_LEFT, ACT_RIGHT):
            self.turn_dir = 1 if a == ACT_LEFT else -1
            self.turn_hold = self.TURN_HOLD
            self.straight_hold = self.STRAIGHT_HOLD
            return 0.0, self.turn_dir * self.w_turn

        if self.turn_hold > 0:
            self.turn_hold -= 1
            return 0.0, self.turn_dir * self.w_turn

        if self.straight_hold > 0:
            self.straight_hold -= 1
            return 0.8 * self.v_max, 0.0

        x, y, th = pose
        dx, dy = target[0] - x, target[1] - y
        dist = math.hypot(dx, dy)
        err = wrap_angle(math.atan2(dy, dx) - th)

        w = max(-self.w_max, min(self.w_max, self.kp * err))
        v = self.v_max * max(0.0, 1.0 - abs(err) / 0.8)
        v = min(v, max(0.03, 1.5 * dist))

        return v, w


# 6. Neural network

class MLP:
    """A fully connected neural network trained using the Adam optimiser."""

    def __init__(self, sizes, rng=None):
        rng = rng or np.random.default_rng(0)
        self.sizes = list(sizes)

        self.W = [rng.normal(0, math.sqrt(2.0 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]

    def forward(self, X):
        """Calculate the network output for the supplied inputs."""
        acts = [np.atleast_2d(X)]

        for k, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W + b

            if k < len(self.W) - 1:
                acts.append(np.maximum(z, 0))
            else:
                z = z - z.max(axis=1, keepdims=True)
                e = np.exp(z)
                acts.append(e / e.sum(axis=1, keepdims=True))

        return acts

    def predict_proba(self, X):
        """Return the predicted probability of each action."""
        return self.forward(X)[-1]

    def predict(self, X):
        """Return the action with the highest predicted probability."""
        return np.argmax(self.predict_proba(X), axis=1)

    def policy(self, f):
        """Choose an action for one robot from its sensor readings."""
        return int(np.argmax(self.forward(np.asarray(f, float))[-1][0]))

    @staticmethod
    def _loss_acc(P, y):
        """Calculate prediction loss and classification accuracy."""
        return float(-np.mean(np.log(P[np.arange(len(y)), y] + 1e-12))), float(np.mean(P.argmax(1) == y))

    def fit(self, Xtr, ytr, Xva, yva, epochs=60, batch=128, lr=3e-3, rng=None, class_weight=None):
        """Train the network and keep the model with the lowest validation loss."""
        rng = rng or np.random.default_rng(1)

        m = [np.zeros_like(p) for p in self.W + self.b]
        v = [np.zeros_like(p) for p in self.W + self.b]

        t = 0
        hist = dict(train_loss=[], val_loss=[], train_acc=[], val_acc=[])
        best, best_state = 1e9, None

        n_cls = self.sizes[-1]
        cw = np.ones(n_cls) if class_weight is None else np.asarray(class_weight, float)

        for ep in range(epochs):
            order = rng.permutation(len(Xtr))

            for i in range(0, len(Xtr), batch):
                idx = order[i:i + batch]
                acts = self.forward(Xtr[idx])

                d = acts[-1].copy()
                d[np.arange(len(idx)), ytr[idx]] -= 1.0
                d *= cw[ytr[idx]][:, None] / len(idx)

                gW, gb = [None] * len(self.W), [None] * len(self.W)

                for k in range(len(self.W) - 1, -1, -1):
                    gW[k] = acts[k].T @ d
                    gb[k] = d.sum(axis=0)

                    if k > 0:
                        d = (d @ self.W[k].T) * (acts[k] > 0)

                t += 1
                params, grads = self.W + self.b, gW + gb

                for j, (p, g) in enumerate(zip(params, grads)):
                    m[j] = 0.9 * m[j] + 0.1 * g
                    v[j] = 0.999 * v[j] + 0.001 * g * g

                    mh, vh = m[j] / (1 - 0.9 ** t), v[j] / (1 - 0.999 ** t)
                    p -= lr * mh / (np.sqrt(vh) + 1e-8)

            ltr, atr = self._loss_acc(self.predict_proba(Xtr), ytr)
            lva, ava = self._loss_acc(self.predict_proba(Xva), yva)

            hist["train_loss"].append(ltr)
            hist["train_acc"].append(atr)
            hist["val_loss"].append(lva)
            hist["val_acc"].append(ava)

            if lva < best:
                best = lva
                best_state = ([w.copy() for w in self.W], [b.copy() for b in self.b])

        if best_state:
            self.W, self.b = best_state

        return hist

    def to_dict(self):
        """Convert the trained network into a dictionary for saving."""
        return dict(sizes=self.sizes, W=[w.tolist() for w in self.W], b=[b.tolist() for b in self.b],
                    ir_baseline=IR_BASELINE, ir_scale=IR_SCALE, actions=ACTIONS)

    @classmethod
    def from_dict(cls, d):
        """Load a network from a saved dictionary."""
        net = cls(d["sizes"])
        net.W = [np.array(w) for w in d["W"]]
        net.b = [np.array(b) for b in d["b"]]
        return net


def make_ir_dataset(n, rng, noise_sigma=15.0, noise_rel=0.03):
    """
    Generate sample sensor readings and the correct avoidance actions.

    Each sample contains zero, one or two simulated obstacles.
    Sensor noise is added to the readings, while the action labels are
    calculated from the clean readings.
    Returns the input features and their corresponding action labels.
    """
    X = np.zeros((n, 8))
    Xc = np.zeros((n, 8))

    for k in range(n):
        d = np.full(8, np.inf)
        n_obs = rng.choice([0, 1, 1, 1, 2])

        for _ in range(n_obs):
            b = rng.uniform(-math.pi, math.pi)
            D = rng.uniform(0.003, 0.09)
            width = rng.uniform(0.3, 1.3)

            dth = np.abs(np.vectorize(wrap_angle)(IR_ANGLES - b))
            seen = dth < width
            di = D * (1.0 + 0.6 * (dth / width) ** 2)

            d = np.where(seen, np.minimum(d, di), d)

        clean = ir_reading_from_distance(np.where(np.isfinite(d), d, 1.0))
        noisy = clean * (1 + rng.normal(0, noise_rel, 8)) + rng.normal(0, noise_sigma, 8)

        Xc[k] = ir_features(clean)
        X[k] = ir_features(np.clip(noisy, 0, 4095))

    y = expert_policy(Xc)

    return X, np.asarray(y, int)