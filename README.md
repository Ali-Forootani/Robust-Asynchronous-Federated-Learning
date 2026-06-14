
# Robust Asynchronous Federated Learning (RAFL) with Non-Convex Client Objectives

This repository implements **Robust Asynchronous Federated Learning (RAFL)** in **pure PyTorch** with **asyncio**—no Flower/PySyft—targeting deep models on **heterogeneous**, **non-IID** clients, optionally with **Byzantine** behavior. RAFL combines:

* **Staleness-aware weighting**,
* **Event-triggered (ET) communication**, and
* **Byzantine-resilient (ASB) aggregation**

with a **non-convex** convergence analysis that quantifies the roles of staleness, heterogeneity, ET bias, and adversarial noise, and proves **sublinear rates with bounded error floors** under standard smoothness/variance assumptions.

> In benign settings, RAFL matches AFL’s efficiency; under adversarial/heterogeneous conditions it maintains accuracy and stability, making it practical for large-scale cross-device deployments.

---

## What's in this repo

* **Main runnable examples (use these):**

  * `async_fdl_mnist_delay_track_lr_non_iid.py` – AFL (vanilla async) on MNIST
  * `async_fdl_fashion_mnist_delay_track_lr_non_iid.py` – AFL on Fashion-MNIST
  * `async_fdl_cifar_delay_track_lr_non_iid.py` – AFL on CIFAR-10
  * `robust_async_fdl_mnist_delay_track_lr_non_iid.py` – **RAFL** on MNIST
  * `robust_async_fdl_fashion_mnist_delay_track_lr_non_iid.py` – **RAFL** on Fashion-MNIST
  * `robust_async_fdl_cifar_delay_track_lr_non_iid.py` – **RAFL** on CIFAR-10
  * `baseline_robust_async_fdl_*.py` – a **baseline robust-async** variant (fixed server stepsize; no ET suppression/ASB refinements)

* **Helper bash launchers on HPC:**
  `asynch_*.sh`, `robust_asynch_*.sh`, `baseline_robust_asynch_*.sh`

* **Plot scripts:**
  `rafl_vs_brafl_vs_asynch_*_plot_final.py`, `compare_rafl_results.py`

* **Data/results:**
  `data/`, `results/`, `results_rafl/`, `training_losses/`, `images/`, plus `extra_modules/` and caches.

---

## Key ideas (theory, briefly)

* **Weighted sampling (without replacement) bound** to control variance with staleness-normalized weights ( \tilde w_c \in [w_{\min}/J,,w_{\max}/J] ).
* **One-step descent inequality** for RAFL (non-convex): quantifies decrease of (L(\theta)) vs. terms from **SGD noise**, **heterogeneity (\nu_t^2)**, **drift (\Xi_t)**, and **Byzantine/ET bias**; enforces admissible stepsizes.
* **Drift bound (\Xi_t)** under L-smoothness, bounded staleness, SGD noise—tightens control near stationarity.
* **Delay/ET-aware schedule** ( \zeta_t = \dfrac{\zeta_0}{\sqrt{t+1}(1+\alpha,\bar\tau_t)(1+\beta,p_t)} ) with safe cap ( \zeta_t \le \frac{1}{4L} ) ensures stability.

For details, see the manuscript sections **II–IV** (notation, lemmas, convergence).

---

## Installation

### Option A — `requirements.txt` (CPU by default; GPU notes below)

```txt
# Core (use matching pair)
torch==2.3.*            # match with torchvision 0.18.*
torchvision==0.18.*

# Numerics & utils
numpy>=1.24,<3
tqdm>=4.66
nest_asyncio>=1.6

# Plotting (headless-safe via Agg in code)
matplotlib>=3.7,<4
pillow>=9.5
```

**GPU wheels (CUDA 12.1 example):**

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.3.* torchvision==0.18.*
```

**CPU wheels:**

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.3.* torchvision==0.18.*
```

### Option B — Conda env (Python 3.10)

```yaml
name: rafl
channels: [pytorch, conda-forge]
dependencies:
  - python=3.10
  - pip
  - numpy>=1.24,<3
  - matplotlib>=3.7,<4
  - pillow>=9.5
  - pip:
      - torch==2.3.*
      - torchvision==0.18.*
      - tqdm>=4.66
      - nest_asyncio>=1.6
```

---

## Reproducibility & HPC stability

The code is designed for **headless/HPC** environments:

* Sets **multiprocessing start method** to **`spawn`** early to avoid deadlocks.
* Uses **`num_workers=0`** for *training* DataLoaders iterated in threads.
* Forces **matplotlib backend `Agg`** (no display needed).
* Seeds NumPy/PyTorch/Python for **determinism**.

These are highlighted in the example code and were crucial to avoid loader deadlocks when mixing threads with DataLoader workers.

---

## Datasets & models

* **MNIST / Fashion-MNIST**: small **CNN** classifier.
* **CIFAR-10**: compact **ResNet** (BasicBlock, 3×3×3).
* **Non-IID partitioning**: **Dirichlet** ( \alpha ) over class indices.

---

## How to run (main examples)

> Tip: Use the `_lr_non_iid.py` files—they’re the canonical, clean runs.

### Asynchronous FL (AFL)

```bash
# MNIST
python async_fdl_mnist_delay_track_lr_non_iid.py

# Fashion-MNIST
python async_fdl_fashion_mnist_delay_track_lr_non_iid.py

# CIFAR-10
python async_fdl_cifar_delay_track_lr_non_iid.py
```

### Robust Asynchronous FL (RAFL)

```bash
# MNIST
python robust_async_fdl_mnist_delay_track_lr_non_iid.py

# Fashion-MNIST
python robust_async_fdl_fashion_mnist_delay_track_lr_non_iid.py

# CIFAR-10
python robust_async_fdl_cifar_delay_track_lr_non_iid.py
```

### Baseline robust-async (no ET schedule/ASB refinements)

```bash
python baseline_robust_async_fdl_mnist_delay_track_lr_non_iid.py
python baseline_robust_async_fdl_fashion_mnist_delay_track_lr_non_iid.py
python baseline_robust_async_fdl_cifar_delay_track_lr_non_iid.py
```

Or use the provided `*.sh` wrappers (AFL / robust / baseline_robust) for quick presets.

---

## Important arguments (common)

| Arg                               | Meaning                            | Typical                                 |
| --------------------------------- | ---------------------------------- | --------------------------------------- |
| `--num_clients`                   | Total clients (C)                  | `10`                                    |
| `--num_rounds`                    | Global rounds (T)                  | `500` (MNIST/F-MNIST), `200` (CIFAR-10) |
| `--local_epochs`                  | Client local steps (I)             | `10`                                    |
| `--clients_per_round`             | Selected per round (J)             | `5`                                     |
| `--batch_size`                    | Local batch size                   | `64`                                    |
| `--alpha`                         | Dirichlet non-IID                  | `0.5`                                   |
| `--eta_client`                    | Client LR ( \eta )                 | e.g., `5e-3`                            |
| `--zeta0`                         | Composite LR seed ( \zeta_0 )      | e.g., `1e-2`                            |
| `--alpha_stale`                   | Staleness weight factor ( \alpha ) | `0.2`                                   |
| `--beta_et`                       | ET factor ( \beta )                | `0.5`                                   |
| `--tau_max_rounds`                | Max simulated staleness            | `3`                                     |
| `--trigger_eps`                   | ET threshold ( \varepsilon )       | `0.0–0.1`                               |
| `--w_min,w_max`                   | Weight clamps                      | `0.5,1.5`                               |
| `--enable_byzantine`              | Turn on attacks                    | `True/False`                            |
| `--byz_frac, byz_mode, byz_scale` | Attack control                     | `0.3, signflip, 10.0`                   |

> The implementation enforces the **safe cap** ( \zeta_t \le \frac{1}{4L} ) from the theory.

---

## Outputs & logging

Each run creates a timestamped directory under `results_rafl/` (or `results/`) with:

* **Per-client losses**: `client_{i}_losses.npy`, `client_{i}_training_loss.png`
* **Server loss proxy (avg client loss per round)**: `server_losses.npy`, `server_training_loss.png`
* **Selected clients per round**: `selected_clients.csv`
* **Execution times**: `execution_times.csv`
* **Theory-linked diagnostics** (RAFL):

  * `tau_bar.png` / series – weighted avg staleness ( \bar\tau_t )
  * `p_t.png` – suppression rate ( p_t )
  * `zeta_t.png` – composite stepsize ( \zeta_t )
  * `gamma_t.png` – server stepsize ( \gamma_t = \zeta_t/(\eta I) )
  * `rho_sb_hat.png` – proxy ( \widehat{\rho}_{SB,t}^2 ) (ASB perturbation)
  * `s_min.png` – honest weight mass lower bound ( s_{\min} )
  * `nu2.npy/png` – heterogeneity proxy ( \hat\nu_t^2 )
  * `test_loss.npy/png`, `test_acc.npy/png` – round-wise generalization

These match the analysis quantities used in the manuscript’s lemmas/recursions.

---

## Repository layout (abridged)

```
.
├── async_fdl_*.py                         # AFL main examples (MNIST/Fashion/CIFAR)
├── robust_async_fdl_*.py                  # RAFL main examples
├── baseline_robust_async_fdl_*.py         # Baseline robust-async examples
├── asynch_*.sh | robust_asynch_*.sh | baseline_robust_asynch_*.sh
├── rafl_vs_brafl_vs_asynch_*_plot_final.py   # Comparison plots
├── compare_rafl_results.py
├── data/ | results/ | results_rafl/ | training_losses/ | images/ | extra_modules/
└── __pycache__/ ...
```

---

## Paper organization (helpful for readers of the code & paper)

* **§II Preliminaries & Assumptions:** notation, client sampling, staleness, ET, robust aggregation; smoothness/variance heterogeneity assumptions.
* **§III Problem Formulation:** async client updates, ET and Byzantine challenges, server rule ( \theta^{t+1}=\theta^t+\gamma_t \hat u^t ).
* **§IV Convergence Analysis:** weighted-sampling variance bound, one-step descent, refined drift, and delay/ET-aware stepsize schedule with sublinear rates and bounded floors.
* **§V Experiments:** MNIST, Fashion-MNIST, CIFAR-10 under heterogeneity and adversaries; RAFL vs AFL vs robust baselines.

---

## Citation

If you use this repository or the accompanying theory, please cite:

```
Ali Forootani, Raffaele Iervolino,
"Robust Asynchronous Federated Learning with Non-Convex Client Objectives", 2025.
```

---

## Contact

* [aliforootani@ieee.org](mailto:aliforootani@ieee.org)
* [aliforootani@gmail.com](mailto:aliforootani@gmail.com)
* [forootani@mpi-magdeburg.mpg.de](mailto:forootani@mpi-magdeburg.mpg.de)
* [forootani@gea.mpg.de](mailto:forootani@gea.mpg.de)

---

## License

**MIT** — see `LICENSE`.

---

**Notes**

* If you want separate CPU/GPU requirements files (or exact pinned versions from a specific HPC run), I can generate `requirements-cpu.txt` and `requirements-gpu-cu121.txt` to match your wheels precisely.

