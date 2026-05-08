import torch
import random
import importlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import sys
import functions.utils as mf
importlib.reload(mf)
from functions.rnn_module import AmmoniaRNN

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Data (identical for both runs)

data = pd.read_csv("data/data.csv")
data = data.rename(columns={"id": "pmid"}).drop(["Unnamed: 0"], axis=1)
data_origin = data.copy()

random.seed(1)
all_pmids   = random.sample(data["pmid"].unique().tolist(), 500)
pmids_val   = all_pmids[:100]
pmids_train = all_pmids[100:]

cat_vars = ["app_method","incorp","till","meas_tech",
            "fer_origin","fer_forme","fer_trt1","fer_trt2","fer_trt3","crop","soil_type"]

cont_vars = ["ct","dt","air_temp","wind_2m","rain_rate","rain_cum",
             "tan_app","app_rate","fer_dm","fer_ph","time_incorp","furrow_z",
             "soil_water","soil_ph","soil_dens","soil_oc","crop_z",
             "air_temp_ind","wind_2m_ind","fer_dm_ind","fer_ph_ind",
             "soil_water_ind","soil_ph_ind","soil_dens_ind","soil_oc_ind",
             "crop_z_ind","time_incorp_ind"]

cat_dims      = [max(data[x]) + 1 for x in cat_vars]
embedding_dims = [8] * len(cat_vars)
input_size     = len(cont_vars) + len(cat_vars)

x_train, y_train = mf.make_tensors(pmids_train, data, "e_cum", cont_vars, cat_vars, DEVICE)
x_val,   y_val   = mf.make_tensors(pmids_val,   data, "e_cum", cont_vars, cat_vars, DEVICE)

results = {}

for nl in ["relu", "tanh"]:
    torch.manual_seed(1)    # same model init for both
    random.seed(0)          # seed 0 batch ordering that causes relu to collapse

    model = AmmoniaRNN(
        input_size     = input_size,
        output_size    = 1,
        hidden_size    = 512,
        nonlinearity   = nl,
        num_layers     = 1,
        bidirectional  = True,
        cat_dims       = cat_dims,
        embedding_dims = embedding_dims,
        ).to(DEVICE)

    fc1_before, fc1_after, fc2_before, fc2_after, gnorm, gnorm_rnn, h_T_norm, rnn_dead = mf.train_model(
        model, num_epochs=5, learning_rate=5e-4,
        x_train=x_train, y_train=y_train, x_val=x_val, DEVICE=DEVICE
       )

    preds = mf.predict_emissions(data_origin, model, pmids_train, cont_vars, cat_vars, "e_cum", DEVICE)
    preds = preds.loc[preds.groupby("pmid")["ct"].idxmax()]

    results[nl] = dict(
        fc1_before  = fc1_before,
        fc1_after   = fc1_after,
        fc2_before  = fc2_before,
        fc2_after   = fc2_after,
        gnorm       = gnorm,
        gnorm_rnn   = gnorm_rnn,
        h_T_norm    = h_T_norm,
        rnn_dead    = rnn_dead,
        preds       = preds,
    )
    print(f"[{nl:4s}]  fc2_after_relu_final = {fc2_after[-1]:.2f}  "
          f"std_pred = {preds['prediction_ecum'].std():.2f}")


# Plots (By Claude Code)
colors = {"relu": "#e74c3c", "tanh": "#2ecc71"}
steps  = range(1, len(results["relu"]["gnorm"]) + 1)

fig = plt.figure(figsize=(24, 12))
fig.suptitle(
    "Controlled experiment: only the RNN nonlinearity changes\n"
    "AmmoniaRNN  |  hidden=512  |  bidirectional  |  same seed, same data, same optimizer\n"
    "fc1(1024→12) → ReLU → fc2(12→6) → ReLU → fc3(6→1)   [head is fixed]",
    fontsize=11, fontweight="bold"
)
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)


# Panel 1 — obs vs pred (relu)
ax = fig.add_subplot(gs[0, 0])
r = results["relu"]["preds"]
ax.scatter(r["e_cum_origin"], r["prediction_ecum"], alpha=0.5, s=12, color=colors["relu"])
ax.set_title("relu RNN - obs vs pred")
ax.set_xlabel("e_cum observed"); ax.set_ylabel("e_cum predicted")

# Panel 2 — obs vs pred (tanh)
ax = fig.add_subplot(gs[0, 1])
r = results["tanh"]["preds"]
ax.scatter(r["e_cum_origin"], r["prediction_ecum"], alpha=0.5, s=12, color=colors["tanh"])
ax.set_title("tanh RNN - obs vs pred")
ax.set_xlabel("e_cum observed"); ax.set_ylabel("e_cum predicted")

# Panel 3 — fc2 after relu (the kill-shot metric)
ax = fig.add_subplot(gs[0, 2])
for nl, h in results.items():
    ax.plot(h["fc2_after"], color=colors[nl], label=nl, lw=2.5)
ax.set_title("sum |h| fc2 after ReLU")
ax.set_xlabel("Training step"); ax.set_ylabel("sum |fc2 output|"); ax.legend(fontsize=10)

# Panel 4 — full signal cascade (relu)
ax = fig.add_subplot(gs[1, 0])
ax.plot(results["relu"]["fc1_before"], label="fc1 before relu", lw=2)
ax.plot(results["relu"]["fc1_after"],  label="fc1 after relu",  lw=2)
ax.plot(results["relu"]["fc2_before"], label="fc2 before relu", lw=2)
ax.plot(results["relu"]["fc2_after"],  label="fc2 after relu",  lw=2)
ax.set_title("relu RNN - signal cascade")
ax.set_xlabel("Training step"); ax.set_ylabel("sum |h|"); ax.legend(fontsize=8)

# Panel 5 — full signal cascade (tanh)
ax = fig.add_subplot(gs[1, 1])
ax.plot(results["tanh"]["fc1_before"], label="fc1 before relu", lw=2)
ax.plot(results["tanh"]["fc1_after"],  label="fc1 after relu",  lw=2)
ax.plot(results["tanh"]["fc2_before"], label="fc2 before relu", lw=2)
ax.plot(results["tanh"]["fc2_after"],  label="fc2 after relu",  lw=2)
ax.set_title("tanh RNN - signal cascade")
ax.set_xlabel("Training step"); ax.set_ylabel("sum |h|"); ax.legend(fontsize=8)

# Panel 6 — gradient norms
ax = fig.add_subplot(gs[1, 2])
for nl, h in results.items():
    gnorm     = [v if np.isfinite(v) else np.nan for v in h["gnorm"]]
    gnorm_rnn = [v if np.isfinite(v) else np.nan for v in h["gnorm_rnn"]]
    ax.semilogy(gnorm,     color=colors[nl], lw=2,   label=f"{nl} — total")
    ax.semilogy(gnorm_rnn, color=colors[nl], lw=1.5, ls="--", label=f"{nl} — RNN only")
ax.set_title("Gradient norms (log)")
ax.set_xlabel("Training step"); ax.set_ylabel("||grad||  (log)"); ax.legend(fontsize=8)

# Panel 7 — dead terminal hidden neurons after training
ax = fig.add_subplot(gs[0, 3])
nls  = list(results.keys())
vals = [results[nl]["rnn_dead"] * 100 for nl in nls]
cols = [colors[nl] for nl in nls]
ax.bar(nls, vals, color=cols, alpha=0.85, width=0.4)
ax.set_title("Dead neurons in h_T after training (%)\n")
ax.set_ylabel("h_T = 0  (%)"); ax.set_ylim(0, 100)

plt.savefig("results/relu_vs_tanh_rnn.png", dpi=200, bbox_inches="tight")
print("\nSaved -> results/relu_vs_tanh_rnn.png")
