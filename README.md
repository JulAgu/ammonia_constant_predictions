# Ammonia Constant Predictions — ReLU Collapse Diagnostic

## The Idea

Using ReLU in the RNN produces a pathological result: the model outputs the same constant for every input, regardless of the data.

The root cause is **dying ReLU** — a forward-pass phenomenon where neurons get permanently zeroed out. Gradient vanishing is not the primary hypothesis but a downstream consequence that locks the collapse in place.

---

## The Math

### 1. Dying ReLU in the recurrence

For a vanilla RNN cell with nonlinearity $f$:

$$h_t = f\!\left(W_h\, h_{t-1} + W_x\, x_t + b\right)$$

With $f = \mathrm{ReLU}$, any neuron $i$ whose pre-activation is non-positive goes to 0:

$$h_t^{(i)} = 0 \quad \text{if} \quad \left[W_h\, h_{t-1} + W_x\, x_t + b\right]_i \leq 0$$

Once dead, the neuron's contribution to the next step vanishes:

$$h_{t+1}^{(i)} = \mathrm{ReLU}\!\left(\underbrace{[W_h]_i \cdot 0}_{=\,0} + [W_x]_i\cdot x_{t+1} + b_i\right)$$

It can only revive if $[W_x]_i \cdot x_{t+1} + b_i > 0$, which becomes increasingly unlikely as weights are updated to fit a dead signal. Neurons accumulate at zero over the sequence.

With $f = \tanh$: output is bounded in $(-1, 1)$ and never exactly zero, so no neuron is permanently silenced.

### 2. Forward signal collapse through the head

Let $h_T \in \mathbb{R}^{1024}$ be the terminal hidden state. The head applies:

$$z_1 = \mathrm{ReLU}(W_1\, h_T + b_1), \quad z_1 \in \mathbb{R}^{12}$$

$$z_2 = \mathrm{ReLU}(W_2\, z_1 + b_2), \quad z_2 \in \mathbb{R}^{6}$$

$$\hat{y} = W_3\, z_2 + b_3, \quad \hat{y} \in \mathbb{R}$$

Each ReLU layer further prunes what survives from the previous one. If $h_T$ carries a weak signal (many dead neurons), $z_1$ is already sparse; $z_2$ may be entirely zero:

$$z_2 = \mathbf{0} \implies \hat{y} = b_3 \quad \forall\, x$$

The model predicts the same constant $b_3$ for every input, this is $\mathrm{std}(\hat{y}) = 0$.

### 3. Gradient vanishing as the "lock-in" mechanism

Because dead neurons have zero derivative, gradients cannot flow back through them during Backpropagation Through Time (BPTT - See https://en.wikipedia.org/wiki/Backpropagation_through_time):

$$\frac{\partial h_t}{\partial h_{t-1}} = \mathrm{diag}\!\left(\mathbf{1}\left[W_h h_{t-1} + W_x x_t + b > 0\right]\right) W_h$$

Write $D_t = \mathrm{diag}\!\left(\mathbf{1}[a_t > 0]\right)$ for the gate matrix at step $t$. BPTT computes the gradient of the loss with respect to an earlier hidden state $h_{t-k}$ by chaining $k$ Jacobians:

$$\frac{\partial \mathcal{L}}{\partial h_{t-k}} = \frac{\partial \mathcal{L}}{\partial h_t} \prod_{j=0}^{k-1} D_{t-j}\, W_h$$

Each factor $D_{t-j} W_h$ zeroes out the rows corresponding to dead neurons at step $t-j$. As $k$ grows, more dead-neuron masks are multiplied together.

When many neurons are dead, $D_{t-j}$ has many zero rows, reducing the effective rank of $W_h$ at each step. If the spectral norm of $D_{t-j} W_h$ is less than 1 — which becomes likely as dead neurons accumulate — the product shrinks **exponentially** with $k$. The gradient signal reaching early timesteps vanishes.

For dead neurons the diagonal entry is 0, blocking the gradient path. The weights that could revive dead neurons receive no update — the collapse is self-reinforcing:

$$\text{dead neurons} \;\longrightarrow\; \nabla_\theta \mathcal{L} \approx 0 \;\longrightarrow\; \text{weights don't change} \;\longrightarrow\; \text{neurons stay dead}$$

Un effet boule de neige effect, quoi.

---

## Experiments

### 1. Controlled comparison — `relu_vs_tanh_rnn.py`

Trains `AmmoniaRNN` twice on the same 400 real sequences, same seed, same optimiser, only the RNN nonlinearity changes.

```bash
uv run python relu_vs_tanh_rnn.py 2>&1 | tee results/relu_vs_tanh_rnn.txt
```

Output: `results/relu_vs_tanh_rnn.png`

### 2. Proof across seeds — `prove_relu_is_the_problem.py`

Trains a toy version of the architecture (no embeddings, synthetic data) across `N_SEEDS` different batch-shuffle seeds for both relu and tanh. Shows the collapse happens for most of seeds.

```bash
uv run python prove_relu_is_the_problem.py 2>&1 | tee results/prove_relu_is_the_problem.txt
```

Output: `results/relu_always_collapses.png`

---

## Setup
Je me suis dis qu'il se peut qu'Armand ne connaisse pas UV, c'est pourquoi j'ai mis ici les commandes pour l'utiliser (très basique)

After cloning the project and in the root (of my branch)
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and create environment
uv sync

# Run the experiments
uv run python relu_vs_tanh_rnn.py
uv run python prove_relu_is_the_problem.py
```