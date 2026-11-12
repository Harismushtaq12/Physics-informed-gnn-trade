# Physics-Informed GNN for Critical Mineral Trade Flows

A physics-informed graph neural network (PI-GNN) that predicts bilateral trade
flows of critical minerals (lithium, cobalt, nickel, rare earths, e-waste)
between countries, and tests whether adding a mass-balance physical
constraint to the training loss improves accuracy and robustness over an
unconstrained GNN and two conventional baselines (a gravity model and
XGBoost).

## The idea, in plain terms

Countries mine lithium, refine it, and ship it to each other. We want to
predict how much moves along each trade route. The trade records are
incomplete — partners disagree, some quantities are withheld, material is
re-routed through intermediaries — so a model has to fill gaps.

A graph neural network does this well until prices move outside the range it
was trained on, and then it fails badly: trained on 2017–2021 and tested on
2024, an unconstrained model predicted 2.4 million tonnes on a route that
actually carried 95 thousand.

The fix tested here is a physical rule added to the training loss: **a
country cannot ship out more metal than it imports plus mines.** The rule has
to be applied carefully, because trade is weighed as rock/chemical compound
while mine output is counted as contained metal, and the two aren't
comparable until converted.

## Headline result

- **Point accuracy (RQ1):** the physics loss did **not** significantly
  improve raw MAE (λ=1.0: 673.76 ± 144.16 vs λ=0: 784.78 ± 242.56, Wilcoxon
  p=0.43) — reported as an honest null result.
- **Mass-balance violation (RQ1):** the physics loss **substantially and
  consistently reduced physical implausibility**, especially in the tail
  (mean violation reduced ~83%, 90th-percentile ~83%, median ~43%).
- Full results, all robustness checks, ablations, and the development log of
  bugs found and fixed along the way are in the notebook itself.

## Repository structure

```
.
├── notebooks/
│   └── PI_GNN_v17_COMPLETE.ipynb   # the full study, run top to bottom
├── data/                           # not tracked — see "Data" below
├── requirements.txt
└── README.md
```

## Data

This notebook does not ship any proprietary data. You need to supply, in
your Google Drive (if running on Colab) or locally (see "Running locally"
below):

| File | Source | Used for |
|---|---|---|
| A BACI trade-data zip (`*BACI*.zip`) | [CEPII BACI](http://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37) | Bilateral trade flows (Step 2–3) |
| Mining production | hardcoded in-notebook (Step 5), literature-sourced, USGS-convention content basis | Node features / RQ2 ground truth |
| `Critical-minerals-extraction.xlsx` (optional) | any independent country × year × material production/reserves panel | Cross-checks the hardcoded mining figures (Step 5, provenance check) |

The OWID CO2/energy dataset (GDP, population, energy use) is pulled
automatically from GitHub in Step 0 — no manual download needed.

### Data sources & references

- **Trade flows.** Gaulier, G. and Zignago, S. (2010). *BACI: International
  Trade Database at the Product-Level. The 1994–2007 Version.* CEPII Working
  Paper, N°2010-23. Data: [CEPII BACI](https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37).
- **Node features (GDP, population, energy use).** Ritchie, H., Rosado, P.,
  and Roser, M. (2023). *CO₂ and Greenhouse Gas Emissions.* Published online
  at [OurWorldInData.org](https://ourworldindata.org/co2-and-greenhouse-gas-emissions).
  Pulled directly from OWID's GitHub CSV in Step 0
  (`owid/co2-data/master/owid-co2-data.csv`).
- **Product classification (HS codes for lithium, cobalt, nickel, rare
  earths, e-waste).** Re-derived from UNCTAD's critical-minerals HS
  classification — [UNCTADstat, Classification of Critical Minerals](https://unctadstat.unctad.org/EN/Classifications.html).
  See Step 3's markdown for the two rounds of code corrections made against
  this source.
- **Mining production convention.** Figures are entered on the same
  "material content" basis USGS reports mineral commodity production in —
  see [USGS Mineral Commodity Summaries](https://www.usgs.gov/centers/national-minerals-information-center/mineral-commodity-summaries)
  (annual). `MINING_CSV` itself (Step 5) is a hand-compiled panel without a
  single traceable citation — flagged as such in the notebook's own
  development log (#16) — which is exactly what the optional
  `Critical-minerals-extraction.xlsx` cross-check (Step 5, provenance
  section) exists to corroborate independently.
- **Conversion factors** (ore/intermediate trade tonnage → contained-metal
  tonnage): lithium carbonate's 18.8% Li-by-mass factor is exact stoichiometry
  (Li₂CO₃). Nickel matte (0.72) and cobalt intermediate (0.90) are
  literature/industry-sourced midpoints, explicitly flagged in the notebook
  (Step 5c) as carrying real uncertainty — see that cell's markdown for the
  specific sources cited for each.

If you reuse this notebook or its numbers, please cite BACI and OWID
directly per their own citation guidance above, in addition to this repo.

## Requirements

```
pip install -r requirements.txt
```

See `requirements.txt` for the package list. `torch_geometric` is installed
from a plain wheel with no compiled extensions (`torch_scatter` /
`torch_sparse` etc. are deliberately **not** required — see Step 4's install
cell for why).

## Running

### On Google Colab (as written)

1. Upload your BACI zip and (optionally) `Critical-minerals-extraction.xlsx`
   to the root of your Google Drive.
2. Open `notebooks/PI_GNN_v17_COMPLETE.ipynb` in Colab.
3. Run top to bottom. **Run Step −1 first, always** — see "Reproducibility"
   below.
4. Full run takes several hours on CPU. Set `QUICK_MODE = True` in the
   experiments section (Steps 21+) for a 5-seed smoke test; use the full 10
   seeds for anything that goes in a paper.

### Locally

Replace the two Google Drive cells (Step 1: mount, Step 2: extract) with
plain local paths to your BACI files and the mining CSV, then run the rest
of the notebook unchanged.

## Reproducibility — read before trusting any GNN number

Two full reruns of this notebook, same code, same fixed seeds (0–9),
produced different results everywhere the GNN was involved — including
flipping the accuracy claim from statistically significant (p=0.0488) to not
(p=0.4316) — while the non-GNN baselines (gravity, XGBoost) were identical
both times.

The cause: `torch.manual_seed()` seeds random-number generation, not
floating-point summation order. Both the physics penalty and the network's
neighbour aggregation use `index_add_`, which adds values into a slot in
whatever order the backend picks — and floating-point addition isn't
associative, so a different order gives a different result in the last bits,
which 150 epochs of gradient descent then amplify.

**Step −1 of the notebook fixes this** (deterministic algorithms, single
thread, all seeds pinned) and must run before any model is trained.

## Roadmap through the notebook

| Steps | What happens |
|---|---|
| −1 to 2 | Deterministic execution, node features, data loading |
| 3 to 5d | Filter trade data, build yearly graphs, add mine production, fix the units problem |
| 6 to 9c | Gravity and gradient-boosting baselines, network architecture, feature standardisation, unconstrained GNN |
| 10 to 12 | The physics-informed model, the λ sweep, edge-masking robustness |
| 13 to 13d | Anomaly screening, explainability, architecture ablation, extra seeds |
| Plots 0–12 | Every figure in the paper |
| 14 to 20 | Multi-year holdout, EV-consumption check, nickel, scrambled-constraint control, scale-free metrics |
| 21 to 24 | Nickel under the 2022 LME price shock, full η sweep, extended λ sweep, balance-only as the main configuration |

The notebook's own development log (final section) records every bug found
and fixed, in order — including two that materially changed the reported
numbers (an HS-code/unit mismatch, and the determinism issue above).

## Author

Mohammad Haris Mushtaq
Muhammad Obaida
Saad Hamid

## License

MIT — see [`LICENSE`](LICENSE).
