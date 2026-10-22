"""
COMPONENT B - Section 3 (cont.): centralised vs distributed decision-making
Same task (allocate 16 weighted sectors to 4 robots), same total PSO budget
(40 particles x 100 iterations), two ways of deciding:
CENTRALISED  - leader (max-ID election) collects positions, runs one swarm,  unicasts the assignment back (star topology, 2(N-1) messages).
DISTRIBUTED  - each robot runs a 10-particle sub-swarm and gossips its best assignment to its two ring neighbours every iteration.

Experiments
  A. Plan level (Monte-Carlo, 100 seeds): coverage and makespan of the plan that
     would really be executed when messages are lost.
  B. Decision latency / compute load.
  C. Closed loop in the kinematic simulator with the trained neural-network
     avoidance policy (20 seeds): success, mission time, collisions.

Run:python B3b_central_vs_distributed.py
"""
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 1. Define HERE so it points directly to the swarm_agent controller folder
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "controllers", "swarm_agent"))
sys.path.insert(0, HERE)

# 2. Now import swarm_core successfully
import swarm_core as sc
from swarm_sim import SwarmSim

OUT = os.path.join(HERE, "..", "outputs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})
N, PARTICLES, ITERS = sc.N_AGENTS, 40, 100
centres = sc.sector_centres()
A_ring = sc.ring_adjacency(N)
RETRIES = 3 # for the 'centralised + retransmissions' variant
results = {}

# A. PLAN LEVEL UNDER MESSAGE LOSS

print("[A] plan quality under message loss")
LOSSES = [0.0, 0.1, 0.2, 0.3, 0.5]
SEEDS = 100
modes = ["centralised", f"centralised + {RETRIES} retries", "distributed"]
stats = {m: {q: dict(cov=[], mk=[], full=[]) for q in LOSSES} for m in modes}
for q in LOSSES:
    for s in range(SEEDS):
        r = np.random.default_rng(10_000 + s)
        runs = {
            "centralised": sc.run_centralised_pso(sc.STARTS, centres, N, PARTICLES, ITERS, np.random.default_rng(r.integers(1 << 31)), loss=q),
            f"centralised + {RETRIES} retries": sc.run_centralised_pso(sc.STARTS, centres, N, PARTICLES, ITERS, np.random.default_rng(r.integers(1 << 31)), loss=q ** (1 + RETRIES)),
            "distributed": sc.run_distributed_pso(sc.STARTS, centres, N, PARTICLES // N, ITERS, A_ring, np.random.default_rng(r.integers(1 << 31)), loss=q),
        }
        for m, res in runs.items():
            met = sc.mission_metrics(sc.executed_sets(res, N), sc.STARTS, centres)
            stats[m][q]["cov"].append(met["coverage"])
            stats[m][q]["mk"].append(met["makespan"])
            stats[m][q]["full"].append(met["coverage"] >= 1.0)
    print("   loss %.1f | " % q + " | ".join(
        "%s cov %.2f full %.0f%%" % (m.split(" ")[0] + ("+r" if "retries" in m else ""),
                                      np.mean(stats[m][q]["cov"]), 100 * np.mean(stats[m][q]["full"])) for m in modes))
results["plan_level"] = {m: {str(q): dict(coverage=float(np.mean(v["cov"])),
                                          full_coverage_rate=float(np.mean(v["full"])),
                                          makespan_mean_s=float(np.mean([x for x, c in zip(v["mk"], v["cov"]) if c >= 1.0] or [np.nan])))
                             for q, v in d.items()} for m, d in stats.items()}


# B. LATENCY AND COMPUTE LOAD
print("[B] latency and compute load")
cost = sc.make_cost_fn(sc.STARTS, centres, N)
probe = np.random.default_rng(0).integers(0, N, len(centres))
t0 = time.perf_counter()
for _ in range(2000):
    cost(probe)
t_eval = (time.perf_counter() - t0) / 2000                       # seconds per fitness evaluation (this PC)
ROUND = 0.1                                                      # assumed radio round period (s)
c = sc.run_centralised_pso(sc.STARTS, centres, N, PARTICLES, ITERS, np.random.default_rng(1))
d = sc.run_distributed_pso(sc.STARTS, centres, N, PARTICLES // N, ITERS, A_ring, np.random.default_rng(1))
lat = {
    "centralised": dict(messages=c["msgs"], evals_on_busiest_robot=c["evals"],
                        compute_s=c["evals"] * t_eval, comm_rounds=3,
                        decision_latency_s=c["evals"] * t_eval + 3 * ROUND),
    "distributed": dict(messages=d["msgs"], evals_on_busiest_robot=d["evals"] // N,
                        compute_s=(d["evals"] // N) * t_eval, comm_rounds=ITERS,
                        decision_latency_s=(d["evals"] // N) * t_eval + ITERS * ROUND),
}
results["latency_and_load"] = dict(eval_time_us=t_eval * 1e6, round_period_s=ROUND, **lat)
for m, v in lat.items():
    print(f"   {m:12s} messages {v['messages']:4d} | evals on busiest robot {v['evals_on_busiest_robot']:5d} | "
          f"latency {v['decision_latency_s']:.2f}s")


#CLOSED LOOP WITH THE NEURAL-NETWORK POLICY

print("[C] closed-loop execution")
wpath = os.path.join(HERE, "..", "controllers", "swarm_agent", "nn_weights.json")
if os.path.exists(wpath):
    with open(wpath) as fh:
        policy = sc.MLP.from_dict(json.load(fh)).policy
else:
    policy = sc.expert_policy
    print("   (nn_weights.json not found - using rule set; run B3 first)")
EXEC_SEEDS = 20
closed = {}
for q in (0.0, 0.3):
    for m in ("centralised", "distributed"):
        rows = []
        for s in range(EXEC_SEEDS):
            r = np.random.default_rng(20_000 + s)
            if m == "centralised":
                res = sc.run_centralised_pso(sc.STARTS, centres, N, PARTICLES, ITERS, r, loss=q)
            else:
                res = sc.run_distributed_pso(sc.STARTS, centres, N, PARTICLES // N, ITERS, A_ring, r, loss=q)
            out = SwarmSim(sc.STARTS, centres, policy, np.random.default_rng(30_000 + s)).run(sc.executed_sets(res, N))
            rows.append(out)
        ok = [o for o in rows if o["success"]]
        closed[f"{m} | loss={q}"] = dict(
            success_rate=float(np.mean([o["success"] for o in rows])),
            mean_coverage=float(np.mean([o["coverage"] for o in rows])),
            mission_time_s=float(np.mean([o["time"] for o in ok])) if ok else None,
            collisions_per_run=float(np.mean([o["collisions"] for o in rows])))
        v = closed[f"{m} | loss={q}"]
        print(f"   loss {q:.1f} | {m:12s} success {v['success_rate']:.0%} | coverage {v['mean_coverage']:.2f} | "
              f"time {v['mission_time_s'] if v['mission_time_s'] is None else round(v['mission_time_s'], 1)} | "
              f"collisions {v['collisions_per_run']:.2f}")
results["closed_loop"] = closed

# FIGURE
cols = {"centralised": "#d95f02", f"centralised + {RETRIES} retries": "#e6ab02", "distributed": "#1b9e77"}
fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
for m in modes:
    ax[0].plot(LOSSES, [np.mean(stats[m][q]["cov"]) for q in LOSSES], marker="o", color=cols[m], label=m)
    ax[1].plot(LOSSES, [np.mean(stats[m][q]["full"]) for q in LOSSES], marker="o", color=cols[m], label=m)
ax[0].set(title="Sectors covered by the executed plan", xlabel="message loss probability", ylabel="coverage fraction", ylim=(0, 1.05))
ax[1].set(title="Runs with complete coverage", xlabel="message loss probability", ylabel="fraction of 100 runs", ylim=(0, 1.05))
ax[0].legend(fontsize=7)
ok_mk = {m: [x for x, cv in zip(stats[m][0.0]["mk"], stats[m][0.0]["cov"]) if cv >= 1.0] for m in ("centralised", "distributed")}
bp = ax[2].boxplot([ok_mk["centralised"], ok_mk["distributed"]], patch_artist=True)
ax[2].set_xticklabels(["centralised", "distributed"])
for patch, col in zip(bp["boxes"], (cols["centralised"], cols["distributed"])):
    patch.set_facecolor(col); patch.set_alpha(0.6)
ax[2].set(title="Plan quality, no message loss (100 seeds)", ylabel="estimated makespan (s) - lower is better")
labels_ = ["centralised", "distributed"]
ax[3].bar(np.arange(2) - 0.2, [lat[m]["decision_latency_s"] for m in labels_], 0.4, color="#7570b3", label="decision latency (s)")
ax3b = ax[3].twinx()
ax3b.bar(np.arange(2) + 0.2, [lat[m]["messages"] for m in labels_], 0.4, color="#66a61e", label="messages")
ax3b.grid(False)
ax[3].set(title="Cost of deciding", xticks=range(2), xticklabels=labels_, ylabel="latency (s)")
ax3b.set_ylabel("messages")
h1, l1 = ax[3].get_legend_handles_labels(); h2, l2 = ax3b.get_legend_handles_labels()
ax[3].legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "B3_4_central_vs_distributed.png"))
plt.close(fig)

with open(os.path.join(OUT, "B3b_central_vs_distributed.json"), "w") as fh:
    json.dump(results, fh, indent=2)
print("done -> outputs/B3_4_central_vs_distributed.png and B3b_central_vs_distributed.json")