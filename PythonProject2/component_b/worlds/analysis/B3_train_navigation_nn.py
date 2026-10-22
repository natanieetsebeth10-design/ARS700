"""
Component B - Section 3: Machine learning for robotic navigation

A small neural network (8 -> 16 -> 16 -> 4) learns obstacle avoidance
from the e-puck's 8 proximity sensors.

Input: 8 normalised IR readings (ps0..ps7)
Output: forward, turn left, turn right, reverse

Pipeline:
1. Generate synthetic sensor data with noise and label it using the
   reference rules. Add real Webots readings if CSV files are available.
2. Compare network architectures and train the selected network using Adam.
3. Evaluate accuracy, precision, recall, F1, the confusion matrix and
   performance at different sensor noise levels.
4. Test 150 random navigation missions against two baseline policies.
5. Save the trained weights for the Webots controller.

Run: python B3_train_navigation_nn.py
The experiments take about 2-3 minutes.
"""

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Add the controller folder so Python can find swarm_core.py.
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "controllers", "swarm_agent"))
sys.path.insert(0, HERE)

# Add the analysis folder so Python can find swarm_sim.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import swarm_core as sc  # noqa: E402
from swarm_sim import single_robot_trial  # noqa: E402

# Set the output and data folders relative to this script.
ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(ANALYSIS_DIR, "..", "outputs"))
DATA = os.path.abspath(os.path.join(ANALYSIS_DIR, "..", "data"))

os.makedirs(OUT, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})
rng = np.random.default_rng(42)
results = {}

# B3.1 Build the dataset
print("[B3.1] building dataset")

# Generate training examples with different sensor noise levels.
chunks = [sc.make_ir_dataset(5000, rng, noise_sigma=s) for s in (10, 25, 50, 80)]
X = np.vstack([c[0] for c in chunks])
y = np.concatenate([c[1] for c in chunks])
n_synth = len(y)

# Add real sensor readings if any Webots CSV files are available.
real = sorted(glob.glob(os.path.join(DATA, "ir_collect_*.csv")))
n_real = 0
if real:
    rows = np.vstack([np.loadtxt(f, delimiter=",", skiprows=1, ndmin=2) for f in real])
    Xr = sc.ir_features(rows[:, :8])
    yr = np.asarray(sc.expert_policy(Xr), int)
    X, y = np.vstack([X, Xr]), np.concatenate([y, yr])
    n_real = len(yr)

# Shuffle the data before splitting it into training, validation and test sets.
perm = rng.permutation(len(y))
X, y = X[perm], y[perm]
n = len(y)
i1, i2 = int(0.70 * n), int(0.85 * n)
Xtr, ytr, Xva, yva, Xte, yte = X[:i1], y[:i1], X[i1:i2], y[i1:i2], X[i2:], y[i2:]

# Save the complete dataset for later use.
np.savetxt(os.path.join(DATA, "ir_dataset.csv"), np.column_stack([X, y]), delimiter=",",
           header=",".join([f"f{i}" for i in range(8)] + ["label"]), comments="", fmt="%.5f")
freq = np.bincount(y, minlength=4) / n
print(f"   {n} samples ({n_synth} synthetic + {n_real} from Webots) | class balance {np.round(freq, 3)}")
results["dataset"] = dict(total=int(n), synthetic=int(n_synth), webots=int(n_real),
                          class_frequency=dict(zip(sc.ACTIONS, freq.round(4).tolist())))
cw = (1.0 / np.sqrt(freq)); cw = cw / cw.mean()  # Give less common classes more weight.

# B3.2 Compare network architectures and train the model
print("[B3.2] architecture sweep and training")
archs = {"8-4": [8, 4], "8-8-4": [8, 8, 4], "8-16-16-4": [8, 16, 16, 4], "8-32-32-4": [8, 32, 32, 4]}
sweep = {}
for name, sizes in archs.items():
    net = sc.MLP(sizes, np.random.default_rng(1))
    h = net.fit(Xtr, ytr, Xva, yva, epochs=40, class_weight=cw, rng=np.random.default_rng(2))
    n_params = sum(w.size for w in net.W) + sum(b.size for b in net.b)
    sweep[name] = dict(val_acc=float(np.max(h["val_acc"])), params=int(n_params))
    print(f"   {name:10s} best val acc {sweep[name]['val_acc']:.4f}  ({n_params} parameters)")
results["architecture_sweep"] = sweep

# Train the selected network and save its weights.
CHOSEN = "8-16-16-4"
net = sc.MLP(archs[CHOSEN], np.random.default_rng(1))
hist = net.fit(Xtr, ytr, Xva, yva, epochs=80, class_weight=cw, rng=np.random.default_rng(2))
with open(os.path.join(HERE, "nn_weights.json"), "w") as fh:
    json.dump(net.to_dict(), fh)

# Plot the training results and compare the architectures.
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
ax[0].plot(hist["train_loss"], label="train"); ax[0].plot(hist["val_loss"], label="validation")
ax[0].set(title="Cross-entropy loss", xlabel="epoch"); ax[0].legend()
ax[1].plot(hist["train_acc"], label="train"); ax[1].plot(hist["val_acc"], label="validation")
ax[1].set(title="Accuracy", xlabel="epoch", ylim=(0.8, 1.0)); ax[1].legend()
names = list(sweep)
ax[2].bar(names, [sweep[k]["val_acc"] for k in names], color=["#bbbbbb", "#bbbbbb", "#1b9e77", "#bbbbbb"])
ax[2].set(title="Architecture sweep (best validation accuracy)", ylim=(min(v["val_acc"] for v in sweep.values()) - 0.02, 1.0))
for tick in ax[2].get_xticklabels():
    tick.set_fontsize(8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "B3_1_learning_curves.png"))
plt.close(fig)

# B3.3 Evaluate the model on the test set
print("[B3.3] test-set evaluation")
pred = net.predict(Xte)
acc = float(np.mean(pred == yte))
cm = np.zeros((4, 4), int)
for t, p in zip(yte, pred):
    cm[t, p] += 1
prec = np.diag(cm) / np.maximum(cm.sum(axis=0), 1)
rec = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
majority = float(np.max(np.bincount(yte)) / len(yte))
rule_noisy = float(np.mean(np.asarray(sc.expert_policy(Xte)) == yte))  # Apply the rules to the noisy readings.
print(f"   NN accuracy {acc:.4f} | majority-class {majority:.4f} | rules on noisy input {rule_noisy:.4f}")
results["test"] = dict(accuracy=acc, majority_baseline=majority, rules_on_noisy_input=rule_noisy,
                       per_class={sc.ACTIONS[k]: dict(precision=float(prec[k]), recall=float(rec[k]), f1=float(f1[k]),
                                                      support=int(cm[k].sum())) for k in range(4)},
                       confusion_matrix=cm.tolist())

# Plot the confusion matrix and the proportion of correct predictions per class.
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
for a_, mat, title, fmt in ((ax[0], cm, "Confusion matrix (test set, counts)", "d"),
                            (ax[1], cm / cm.sum(axis=1, keepdims=True), "Row-normalised (recall per class)", ".2f")):
    im = a_.imshow(mat, cmap="Blues")
    a_.set(title=title, xlabel="predicted", ylabel="true", xticks=range(4), yticks=range(4),
           xticklabels=sc.ACTIONS, yticklabels=sc.ACTIONS)
    a_.grid(False)
    for i in range(4):
        for j in range(4):
            a_.text(j, i, format(mat[i, j], fmt), ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() * 0.55 else "black")
    fig.colorbar(im, ax=a_, fraction=0.046)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "B3_2_confusion_matrix.png"))
plt.close(fig)

# Check how sensor noise affects the neural network and rule-based policy.
sigmas = [0, 15, 40, 80, 120, 200]
acc_nn, acc_rule = [], []
for s in sigmas:
    Xn, yn = sc.make_ir_dataset(3000, np.random.default_rng(900 + s), noise_sigma=max(s, 1e-9), noise_rel=0.03 if s else 0.0)
    acc_nn.append(float(np.mean(net.predict(Xn) == yn)))
    acc_rule.append(float(np.mean(np.asarray(sc.expert_policy(Xn)) == yn)))
results["accuracy_vs_noise"] = dict(sigma=sigmas, nn=acc_nn, rules=acc_rule)

# B3.4 Test the policies in closed-loop navigation missions
print("[B3.4] closed-loop evaluation (150 missions per policy and noise level)")
policies = {"neural network": net.policy,
            "rule set on noisy IR": lambda f: sc.expert_policy(f),
            "no avoidance": lambda f: 0}
TRIALS = 150
closed = {}
for sigma in (15, 60):
    for name, pol in policies.items():
        runs = [single_robot_trial(pol, np.random.default_rng(5000 + k), noise_sigma=sigma) for k in range(TRIALS)]
        succ = float(np.mean([r["success"] for r in runs]))
        coll = float(np.mean([r["collisions"] for r in runs]))
        tmean = float(np.mean([r["time"] for r in runs if r["reached"]] or [np.nan]))
        reward = float(np.mean([10.0 * r["reached"] - 5.0 * r["collisions"] - 0.05 * r["time"] for r in runs]))
        closed[f"{name} | sigma={sigma}"] = dict(success_rate=succ, collisions_per_mission=coll,
                                                 mean_time_s=tmean, mean_reward=reward)
        print(f"   sigma {sigma:3d} | {name:22s} success {succ:5.1%} | collisions/mission {coll:.2f} | reward {reward:6.2f}")
results["closed_loop"] = closed

# Plot accuracy, mission success and collisions for the different policies.
fig, ax = plt.subplots(1, 3, figsize=(13, 3.7))
ax[0].plot(sigmas, acc_nn, marker="o", label="neural network")
ax[0].plot(sigmas, acc_rule, marker="s", label="rule set on noisy IR")
ax[0].set(title="Accuracy vs IR noise", xlabel="noise sigma (sensor counts)", ylabel="accuracy"); ax[0].legend()
labels = list(policies)
w_ = 0.38
for k, sigma in enumerate((15, 60)):
    ax[1].bar(np.arange(3) + (k - 0.5) * w_, [closed[f"{p} | sigma={sigma}"]["success_rate"] for p in labels], w_,
              label=f"noise sigma {sigma}")
    ax[2].bar(np.arange(3) + (k - 0.5) * w_, [closed[f"{p} | sigma={sigma}"]["collisions_per_mission"] for p in labels], w_,
              label=f"noise sigma {sigma}")
for a_, t, yl in ((ax[1], "Mission success rate", "fraction"), (ax[2], "Collisions per mission", "count")):
    a_.set(title=t, ylabel=yl, xticks=range(3), xticklabels=[l.replace(" ", "\n", 1) for l in labels]); a_.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "B3_3_closed_loop.png"))
plt.close(fig)

# Save the evaluation results as JSON.
with open(os.path.join(OUT, "nn_weights.json"), "w") as fh:
    json.dump(results, fh, indent=2)
print("saved weights -> controllers/swarm_agent/nn_weights.json ; results -> outputs/nn_weights.json")
