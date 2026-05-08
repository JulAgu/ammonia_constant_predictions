import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

N_SEEDS = 100

def make_toy_sequences(n=300, seq_len=20, input_dim=30, seed=42, device="cpu"):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, seq_len, input_dim)).astype(np.float32) * 3.0
    Y_raw = np.abs(X[:, :, 0]).cumsum(axis=1).astype(np.float32)
    Y = (Y_raw - Y_raw.mean()) / (Y_raw.std() + 1e-8)
    X_t = torch.tensor(X).permute(1, 0, 2).to(device)
    Y_t = torch.tensor(Y).unsqueeze(-1).to(device)
    split = int(n * 0.8)
    return X_t[:, :split], Y_t[:split], X_t[:, split:], Y_t[split:]


# As AmmoniaRNN head but whitout embeddings

class AmmoniaRNNToy(nn.Module):
    def __init__(self, input_dim=30, hidden_size=512, rnn_nonlinearity="relu"):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_size, num_layers=1,
                          nonlinearity=rnn_nonlinearity, bidirectional=True,
                          batch_first=False)
        self.fc1 = nn.Linear(hidden_size * 2, 12)
        self.fc2 = nn.Linear(12, 6)
        self.fc3 = nn.Linear(6, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        h, _  = self.rnn(x)
        post1 = self.relu(self.fc1(h))
        post2 = self.relu(self.fc2(post1))
        pred  = self.fc3(post2)
        return pred, h, post2


def train(model, X, Y, X_val, Y_val, shuffle_seed, lr=1e-2, epochs=60, batch_size=32):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    n = Y.shape[0]
    rng = torch.Generator()
    rng.manual_seed(shuffle_seed)   # only controls batch order

    fc2_after_history = []

    for _ in range(epochs):
        model.train()
        idx = torch.randperm(n, generator=rng)
        for i in range(0, n, batch_size):
            bi = idx[i:i+batch_size]
            xb, yb = X[:, bi], Y[bi].permute(1, 0, 2)
            optimizer.zero_grad()
            pred, _, _ = model(xb)
            loss = ((pred - yb) ** 2).mean()
            if not torch.isfinite(loss):
                continue
            loss.backward()
            gn = sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None) ** 0.5
            if torch.isfinite(gn):
                optimizer.step()

        model.eval()
        with torch.no_grad():
            _, _, post2_val = model(X_val)
            fc2_after_history.append(post2_val.abs().mean().item())

    model.eval()
    with torch.no_grad():
        pred_val, h_val, _ = model(X_val)
        std_pred = pred_val.std().item() if torch.isfinite(pred_val).all() else float("nan")
        rnn_dead = (h_val.abs() < 1e-6).float().mean().item()
        val_mse  = ((pred_val - Y_val.permute(1, 0, 2)) ** 2).mean().item()
        h_T      = torch.cat([h_val[-1, :, :512], h_val[0, :, 512:]], dim=-1)
        h_T_norm = h_T.norm(dim=-1).mean().item()

    return fc2_after_history, h_T_norm, std_pred, rnn_dead, val_mse


X_tr, Y_tr, X_val, Y_val = make_toy_sequences(device=DEVICE)

results = {"relu": [], "tanh": []}

for shuffle_seed in range(N_SEEDS):
    for nl in ["relu", "tanh"]:
        torch.manual_seed(1)          # identical model init every run
        model = AmmoniaRNNToy(rnn_nonlinearity=nl).to(DEVICE)
        fc2_hist, h_T_norm, std_pred, rnn_dead, val_mse = train(
            model, X_tr, Y_tr, X_val, Y_val, shuffle_seed=shuffle_seed
        )
        results[nl].append(dict(fc2_hist=fc2_hist, h_T_norm=h_T_norm, std_pred=std_pred, rnn_dead=rnn_dead, val_mse=val_mse))
        print(f"  [{nl:4s} shuffle={shuffle_seed}]  "
              f"fc2_after={fc2_hist[-1]:.4f}  std_pred={std_pred:.4f}  "
              f"rnn_dead={rnn_dead:.1%}  val_mse={val_mse:.4f}  h_T_norm={h_T_norm:.4f}")


def _arr(nl, key):
    return np.array([r[key] for r in results[nl]], dtype=float)

relu_std_pred = _arr("relu", "std_pred")
tanh_std_pred = _arr("tanh", "std_pred")

print("\n" + "=" * 72)
print(f"SUMMARY STATISTICS  (N_SEEDS = {N_SEEDS})")
print(f"{'Metric':<24} {'relu  mean ± std':>22} {'tanh  mean ± std':>22}")
print("-" * 72)

_metrics = [
    ("fc2_after_final",  "fc2_hist", lambda r: r["fc2_hist"][-1],  "{:.4f}"),
    ("std_pred",         "std_pred", lambda r: r["std_pred"],       "{:.4f}"),
    ("rnn_dead (%)",     "rnn_dead", lambda r: r["rnn_dead"] * 100, "{:.1f}"),
    ("val_mse",          "val_mse",  lambda r: r["val_mse"],        "{:.4f}"),
    ("h_T_norm",         "h_T_norm", lambda r: r["h_T_norm"],       "{:.4f}"),
]
for name, _, fn, fmt in _metrics:
    rv = np.array([fn(r) for r in results["relu"]])
    tv = np.array([fn(r) for r in results["tanh"]])
    rs = f"{fmt.format(np.nanmean(rv))} ± {fmt.format(np.nanstd(rv))}"
    ts = f"{fmt.format(np.nanmean(tv))} ± {fmt.format(np.nanstd(tv))}"
    print(f"  {name:<22} {rs:>22} {ts:>22}")

relu_collapsed = int((relu_std_pred < 1e-3).sum())
tanh_collapsed = int((tanh_std_pred < 1e-3).sum())
print(f"\n  Collapse rate (std_pred < 1e-3):  "
      f"relu = {relu_collapsed}/{N_SEEDS}  |  tanh = {tanh_collapsed}/{N_SEEDS}")

print("=" * 72 + "\n")


# Plots (By Claude Code)

colors = {"relu": "#e74c3c", "tanh": "#2ecc71"}
epochs_x = np.arange(1, 61)
seeds_x  = np.arange(N_SEEDS)
width    = 0.35

fig, axes = plt.subplots(1, 4, figsize=(24, 6))
fig.suptitle(
    "Proof: ReLU in the RNN causes collapse regardless of data shuffle\n"
    "Same model init (torch.manual_seed=1)"
    "Archi: BidirRNN(hidden=512, relu|tanh) -> fc1(1024->12) -> ReLU -> fc2(12->6) -> ReLU -> fc3(6->1)",
    fontsize=11, fontweight="bold"
)

# Panel 1 — fc2 after relu signal over training (all seeds)
ax = axes[0]
for nl in ["relu", "tanh"]:
    for r in results[nl]:
        ax.plot(epochs_x, r["fc2_hist"], color=colors[nl], alpha=0.3, lw=1.5)
for nl in ["relu", "tanh"]:
    mean_curve = np.nanmean([r["fc2_hist"] for r in results[nl]], axis=0)
    ax.plot(epochs_x, mean_curve, color=colors[nl], lw=3.5, label=f"{nl}  (mean)")
ax.set_title("fc2 after ReLU signal over training")
ax.set_xlabel("Epoch"); ax.set_ylabel("Mean |fc2 output|"); ax.legend(fontsize=10)

# Panel 2 — std(predictions) per seed
ax = axes[1]
relu_std = [r["std_pred"] for r in results["relu"]]
tanh_std = [r["std_pred"] for r in results["tanh"]]
ax.bar(seeds_x - width/2, relu_std, width, color=colors["relu"], label="relu", alpha=0.85)
ax.bar(seeds_x + width/2, tanh_std, width, color=colors["tanh"], label="tanh", alpha=0.85)
ax.axhline(0, color="black", lw=0.8, ls="--")
ax.set_title("Std of predictions per shuffle seed")
ax.set_xlabel("Shuffle seed"); ax.set_ylabel("Std(predictions)")
ax.set_xticks(seeds_x); ax.legend(fontsize=10)

# Panel 3 — dead RNN neurons per seed
ax = axes[2]
relu_dead = [r["rnn_dead"] * 100 for r in results["relu"]]
tanh_dead = [r["rnn_dead"] * 100 for r in results["tanh"]]
ax.bar(seeds_x - width/2, relu_dead, width, color=colors["relu"], label="relu", alpha=0.85)
ax.bar(seeds_x + width/2, tanh_dead, width, color=colors["tanh"], label="tanh", alpha=0.85)
ax.set_title("Dead RNN neurons per shuffle seed (%)")
ax.set_xlabel("Shuffle seed"); ax.set_ylabel("h_t = 0  (%)")
ax.set_xticks(seeds_x); ax.legend(fontsize=10)
ax.set_ylim(0, 100)

# Panel 4 — h_T norm on val set after training, per seed
ax = axes[3]
relu_h_T = [r["h_T_norm"] for r in results["relu"]]
tanh_h_T = [r["h_T_norm"] for r in results["tanh"]]
ax.bar(seeds_x - width/2, relu_h_T, width, color=colors["relu"], label="relu", alpha=0.85)
ax.bar(seeds_x + width/2, tanh_h_T, width, color=colors["tanh"], label="tanh", alpha=0.85)
ax.set_title("Mean ||h_T||₂ on val set after training")
ax.set_xlabel("Shuffle seed"); ax.set_ylabel("Mean ||h_T||₂")
ax.set_xticks(seeds_x); ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig("results/relu_always_collapses.png", dpi=200, bbox_inches="tight")
print("\nSaved -> results/relu_always_collapses.png")
