"""
Component B - Sections 1 and 2: simulation setup, consensus and swarm intelligence

Offline Matplotlib experiments to support the Webots simulations:
B2.1 Average consensus: convergence on a ring and the effect of network topology
B2.2 Leader election: rounds needed for different team sizes and topologies
B2.3 Discrete PSO for task allocation: convergence and final allocation map
B2.4 Group behaviour: robot paths and coverage over time

Run: python B1_B2_consensus_pso.py
The experiments take about 1-2 minutes.
Output files are saved in ../outputs/
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Go up from worlds/analysis to component_b
COMPONENT_B = os.path.dirname(os.path.dirname(HERE))

sys.path.insert(0, HERE)
sys.path.insert(
    0, os.path.join(COMPONENT_B, "controllers", "swarm_agent")
)

import swarm_core as sc  # noqa: E402
from swarm_sim import SwarmSim  # noqa: E402

OUT = os.path.join(COMPONENT_B, "outputs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 9
})

AGENT_COLOURS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
N = sc.N_AGENTS
centres = sc.sector_centres()
results = {}

# B2.1 Average consensus
print("[B2.1] average consensus")

# Each agent starts with a different estimate. Consensus helps all agents
# reach the same average value.
x0 = np.array([12.0, 3.0, 8.0, 21.0])
topologies = {"ring": sc.ring_adjacency(N), "path": sc.path_adjacency(N),
              "star": sc.star_adjacency(N), "fully connected": sc.full_adjacency(N)}
ITERS = 25
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
hist = sc.average_consensus(x0, topologies["ring"], ITERS)
for i in range(N):
    ax[0].plot(hist[:, i], color=AGENT_COLOURS[i], marker="o", ms=3, label=f"agent {i}")
ax[0].axhline(x0.mean(), color="k", ls="--", lw=1, label=f"true mean = {x0.mean():.1f}")
ax[0].set(title="Average consensus on a ring (Metropolis weights)", xlabel="round k", ylabel="agent state")
ax[0].legend(fontsize=7)

rates = {}
for name, A in topologies.items():
    W = sc.metropolis_weights(A)
    h = sc.average_consensus(x0, A, ITERS)
    err = np.max(np.abs(h - x0.mean()), axis=1)
    rates[name] = sc.consensus_rate(W)
    ax[1].semilogy(np.maximum(err, 1e-16), marker=".", label=f"{name} (rate {rates[name]:.2f})")
ax[1].set(title="Convergence vs topology", xlabel="round k", ylabel="max |x_i - mean|")
ax[1].legend(fontsize=7)

rng = np.random.default_rng(0)
for q in (0.0, 0.3, 0.6):
    errs = []
    for rep in range(30):
        h = sc.average_consensus(x0, topologies["ring"], ITERS, link_drop=q, rng=rng)
        errs.append(np.max(np.abs(h - x0.mean()), axis=1))
    ax[2].semilogy(np.maximum(np.mean(errs, axis=0), 1e-16), label=f"link failure {int(q * 100)}%")
ax[2].set(title="Robustness to link failures (ring, mean of 30)", xlabel="round k", ylabel="max |x_i - mean|")
ax[2].legend(fontsize=7)
fig.tight_layout()
fig.savefig(
    os.path.join(OUT, "B2_1_consensus.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close(fig)
results["consensus_rate_by_topology"] = rates

# Find how many rounds the ring needs to get within 1% of the initial spread.
h = sc.average_consensus(x0, topologies["ring"], 200)
spread0 = np.max(np.abs(x0 - x0.mean()))
results["ring_rounds_to_1pct"] = int(np.argmax(np.max(np.abs(h - x0.mean()), axis=1) < 0.01 * spread0))

print("   rates", {k: round(v, 3) for k, v in rates.items()}, "| ring rounds to 1%:", results["ring_rounds_to_1pct"])

# B2.2 Leader election
print("[B2.2] leader election")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
hle, agree = sc.leader_election(sc.ring_adjacency(N))
im = ax[0].imshow(hle.T, aspect="auto", cmap="viridis")
ax[0].set(title=f"Max-ID flooding, ring of {N}: agreed after {agree} rounds",
          xlabel="round", ylabel="agent", yticks=range(N))
ax[0].grid(False)
fig.colorbar(im, ax=ax[0], label="leader ID believed")
sizes = list(range(3, 21))
for name, f in (("ring", sc.ring_adjacency), ("path", sc.path_adjacency), ("fully connected", sc.full_adjacency)):
    ax[1].plot(sizes, [sc.leader_election(f(n), max_rounds=n)[1] for n in sizes], marker="o", ms=3, label=name)
ax[1].set(title="Rounds to elect a leader vs team size", xlabel="number of agents", ylabel="rounds")
ax[1].legend()
fig.tight_layout()
fig.savefig(
    os.path.join(OUT, "B2_2_leader_election.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close(fig)
results["leader_election_rounds_ring4"] = agree

# B2.3 PSO task allocation
print("[B2.3] PSO task allocation")
ITERS_PSO, PARTICLES = 100, 40  # Both modes use the same total number of particles
A_ring = sc.ring_adjacency(N)
cost = sc.make_cost_fn(sc.STARTS, centres, N)
greedy = sc.greedy_assignment(sc.STARTS, centres, N)
greedy_cost = cost(greedy)

# Use simulated annealing with restarts to find a reference solution.
# This reference is only used for comparison, not by the swarm.
rng = np.random.default_rng(0)
ref_best, ref_assign = 1e9, None
for r in range(30):
    a = rng.integers(0, N, len(centres)); f = cost(a); T = 3.0
    for _ in range(4000):
        b = a.copy(); b[rng.integers(len(centres))] = rng.integers(N); fb = cost(b)
        if fb < f or rng.random() < np.exp((f - fb) / T):
            a, f = b, fb
        T *= 0.9985
        if f < ref_best:
            ref_best, ref_assign = f, a.copy()

SEEDS = 30
cent_hist, dist_hist, dist_agree = [], [], []
for s in range(SEEDS):
    r = np.random.default_rng(1000 + s)
    c = sc.run_centralised_pso(sc.STARTS, centres, N, PARTICLES, ITERS_PSO, r)
    d = sc.run_distributed_pso(sc.STARTS, centres, N, PARTICLES // N, ITERS_PSO, A_ring, r)
    cent_hist.append(c["hist"])
    dist_hist.append(d["hist"].min(axis=1))
    dist_agree.append(len({tuple(a) for a in d["assigns"]}) == 1)
cent_hist, dist_hist = np.array(cent_hist), np.array(dist_hist)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
for name, H, col in (("centralised (1 swarm x 40 particles)", cent_hist, "#d95f02"),
                     ("distributed (4 sub-swarms x 10, ring gossip)", dist_hist, "#1b9e77")):
    m, sd = H.mean(axis=0), H.std(axis=0)
    ax[0].plot(m, color=col, label=name)
    ax[0].fill_between(range(len(m)), m - sd, m + sd, color=col, alpha=0.2)
ax[0].axhline(greedy_cost, color="grey", ls=":", label=f"nearest-agent heuristic ({greedy_cost:.1f})")
ax[0].axhline(ref_best, color="k", ls="--", label=f"annealing reference ({ref_best:.1f})")
ax[0].set(title=f"PSO convergence (mean ± sd, {SEEDS} seeds)", xlabel="iteration", ylabel="plan cost = makespan estimate (s)")
ax[0].legend(fontsize=7)

# Plot the best distributed plan as a map of sector assignments.
r = np.random.default_rng(1000)
d = sc.run_distributed_pso(sc.STARTS, centres, N, PARTICLES // N, ITERS_PSO, A_ring, r)
best_i = int(np.argmin([cost(a) for a in d["assigns"]]))
plan = d["assigns"][best_i]
half = sc.ARENA / 2
ax[1].set(xlim=(-half, half), ylim=(-half, half), aspect="equal", title="Allocation found by distributed PSO")
for s, (cx, cy) in enumerate(centres):
    a = int(plan[s])
    ax[1].add_patch(plt.Rectangle((cx - 0.25, cy - 0.25), 0.5, 0.5, color=AGENT_COLOURS[a], alpha=0.35,
                                  hatch="//" if sc.SECTOR_WORK[s] > 5 else None, ec="k", lw=0.5))
    ax[1].text(cx, cy, f"{s}\n{sc.SECTOR_WORK[s]:.0f}s", ha="center", va="center", fontsize=7)
for (ox, oy, h) in sc.OBSTACLES:
    ax[1].add_patch(plt.Rectangle((ox - h, oy - h), 2 * h, 2 * h, color="k"))
for i, st in enumerate(sc.STARTS):
    ax[1].plot(st[0], st[1], marker="*", ms=13, color=AGENT_COLOURS[i], mec="k")
    ax[1].text(st[0], st[1] - 0.12, f"A{i}", ha="center", fontsize=8)
ax[1].grid(False)
fig.tight_layout()
fig.savefig(
    os.path.join(OUT, "B2_3_pso_allocation.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close(fig)

times_plan = sc.agent_times(plan, sc.STARTS, centres, N)
results["pso"] = dict(
    iterations=ITERS_PSO, particles_total=PARTICLES,
    central_final_mean=float(cent_hist[:, -1].mean()), central_final_sd=float(cent_hist[:, -1].std()),
    distributed_final_mean=float(dist_hist[:, -1].mean()), distributed_final_sd=float(dist_hist[:, -1].std()),
    greedy_cost=float(greedy_cost), annealing_reference=float(ref_best),
    distributed_agents_agree_fraction=float(np.mean(dist_agree)),
    plan_agent_times_s=[round(t, 1) for t in times_plan], plan=[int(a) for a in plan])
print("   central %.1f±%.1f | distributed %.1f±%.1f | greedy %.1f | reference %.1f | agree %.0f%%" % (
    cent_hist[:, -1].mean(), cent_hist[:, -1].std(), dist_hist[:, -1].mean(), dist_hist[:, -1].std(),
    greedy_cost, ref_best, 100 * np.mean(dist_agree)))


# B2.4 Group behaviour
# Run the distributed plan in the simulator using the obstacle-avoidance policy.
print("[B2.4] emergent behaviour (kinematic execution of the distributed plan)")
wpath = os.path.join(HERE, "..", "controllers", "swarm_agent", "nn_weights.json")
if os.path.exists(wpath):
    with open(wpath) as fh:
        policy = sc.MLP.from_dict(json.load(fh)).policy
    policy_name = "trained neural network"
else:
    policy, policy_name = sc.expert_policy, "expert rules (run B3 first for the neural net)"
sets = [[s for s in range(len(plan)) if plan[s] == i] for i in range(N)]
sim = SwarmSim(sc.STARTS, centres, policy, np.random.default_rng(5))
out = sim.run(sets)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].set(xlim=(-half, half), ylim=(-half, half), aspect="equal",
          title=f"Executed trajectories ({policy_name})")
for (ox, oy, h) in sc.OBSTACLES:
    ax[0].add_patch(plt.Rectangle((ox - h, oy - h), 2 * h, 2 * h, color="k"))
for s, (cx, cy) in enumerate(centres):
    ax[0].add_patch(plt.Rectangle((cx - 0.25, cy - 0.25), 0.5, 0.5, fill=False, ec="grey", lw=0.5))
for i, p in enumerate(out["paths"]):
    p = np.array(p)
    ax[0].plot(p[:, 0], p[:, 1], color=AGENT_COLOURS[i], lw=1.5, label=f"agent {i}")
    ax[0].plot(*p[0], marker="*", ms=11, color=AGENT_COLOURS[i], mec="k")
ax[0].legend(fontsize=7, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.06))
ax[0].grid(False)

# Compare the execution time of the PSO plan with the nearest-agent plan.
o_greedy = SwarmSim(sc.STARTS, centres, policy, np.random.default_rng(5)).run(
    [[s for s in range(len(greedy)) if greedy[s] == i] for i in range(N)])
bars = ax[1].bar(["nearest-agent\nheuristic", "distributed\nPSO plan"], [o_greedy["time"], out["time"]],
                 color=["grey", "#1b9e77"])
for bar, v in zip(bars, [o_greedy["time"], out["time"]]):
    ax[1].text(bar.get_x() + bar.get_width() / 2, v + 0.5, f"{v:.1f} s", ha="center", fontsize=9)
ax[1].set(title="Time to cover all 16 sectors (kinematic execution)", ylabel="mission time (s)")
fig.tight_layout()
fig.savefig(
    os.path.join(OUT, "B2_4_emergent_behaviour.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close(fig)
results["execution"] = dict(policy=policy_name, mission_time_s=out["time"], success=bool(out["success"]),
                            collisions=out["collisions"], greedy_mission_time_s=o_greedy["time"])
print("   PSO plan time %.1fs, collisions %d | greedy plan time %.1fs" % (
    out["time"], out["collisions"], o_greedy["time"]))

with open(os.path.join(OUT, "B1_B2_results.json"), "w") as fh:
    json.dump(results, fh, indent=2)
print("done -> outputs/B1_B2_results.json and 4 figures")