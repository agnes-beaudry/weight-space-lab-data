# Weight-Space Network Lab: data

Data from the Math 5810 *Weight-Space Network Lab* page, which reruns the experiments of
[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/).
The lab page itself is `index.html`, served at https://agnes-beaudry.github.io/weight-space-lab-data/ (no login needed).
Everything here was extracted from the published lab page, so it is exactly what the page shows.

Each experiment has three sources of answers:

- `paper`: Astra, the paper's own run
- `astra`: our re-run of Astra
- `claude`: Claude Opus 5.5

## Layout

| Folder | What it holds |
|---|---|
| `exp1-which-game/` | Four PPO policy networks (Breakout, Pong, Pac-Man, Freeway on a 10×10 grid). `weights/` has the full actor and critic weights. `runs.csv` gives seed, training steps, mean return, action usage and each model's guess. The folder also has the prompts (weights as text), the full answers and the claim checks. |
| `exp1-v2-no-labels/` | Exp 1 again with no labels: the same four networks and weights, but the prompt gives only the architecture and the numbers. Nothing says the inputs are a 10×10 grid with 5 channels, what the 5 outputs are, how it was trained, or that it plays a game. `prompts/` has the exact prompts, `answers/` the replies (Astra at high and at xhigh, Claude Opus 5.5 at high), `claim_checks/` the checked claims, and `runs.csv` the thinking tokens and what each answer worked out: the input layout, the channel roles, the outputs and the game. |
| `exp2-which-shape/` | Three 12×12 DDPMs (circles, squares, triangles). `weights/` has the full weights. `run_details/` has the loss curves and generated samples. `runs.csv` gives a classifier's verdict on the samples and each model's description. The folder also has the prompts, answers and claim checks. |
| `exp3-which-language/` | A 4-layer, 6-head character transformer trained on Python, romaji or English. `head_metrics.csv` gives the attention statistics per head. `analysis_code/` has every code call each model ran, with its output. `answers/` has the final answers (2 runs per source). `runs_and_audit.csv` gives the rule-audit result for each run. |
| `exp4-fix-network/` | A failing car-control policy and the edits each run made to fix it. `runs.csv` gives the outcome over 1,024 test episodes. `edits/` lists the weight changes. `weight_heatmaps.json` holds the base weight matrices, quantized for display; each matrix lists its scale, and biases are not included. The folder also has the answers and claim checks. |
| `exp5-write-a-player/` | MinAtar players written from all-zero weights in one reply (Breakout, Asterix, Seaquest). `scores.csv` compares mean scores with PPO at 100M and 500M steps. The folder also has the full answers (which include the `edits` blocks), the sample game traces and the claim checks. |
| `raw/` | The page's embedded data objects, verbatim as JSON (`NETS`, `EXPL`, `E1P` … `E5A`, plus small label tables in `lab_tables.json`). Use this if a derived file seems to be missing something. |

## Weight format

Weight files are JSON objects of the form `{"tensor name": {"shape": [...], "data": [...]}}`. The `data` array is row-major and rounded to 4 decimals. To load one in Python:

```python
import json, numpy as np
net = json.load(open("exp1-which-game/weights/breakout_a9f535cc.json"))
W1 = np.array(net["actor/dense1/kernel"]["data"]).reshape(net["actor/dense1/kernel"]["shape"])
```

## Not included

The page does not contain these, so they are not here:

- the Exp 3 transformer weights (only their per-head statistics)
- the exact Exp 4 base policy weights (only quantized matrices, without biases, and the edit lists)
