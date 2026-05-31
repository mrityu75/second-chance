# Second Chance

> Every rejected kidney is a prediction about the future.

A probability experience about uncertainty, built for the Stanford CS109 Probability Challenge.

When a donor kidney becomes available, a transplant center has under an hour to decide:
accept it now, or reject it and wait for a potentially better offer. Nobody knows what the
future holds. **Second Chance** lets you step into that decision, then watch the thousands of
possible futures it creates — drawn from real UNOS kidney offer data.

## The experience

Five screens take you from gut decision to probabilistic insight:

1. **The hook** — one patient, one kidney, one choice. No numbers. Just judgment.
2. **The Future Galaxy** — 8,000 simulated futures explode into view. Green = transplanted,
   blue = still waiting, red = died waiting.
3. **Parallel universes** — Accept vs. Wait, side by side, with outcome distributions.
4. **The probability landscape** — how often each choice wins, with a 95% confidence interval,
   and a Thompson-sampling recommendation.
5. **The hidden variable** — information theory reveals what actually drives the outcome.
   (Spoiler: it's not the kidney.)

## The probability (CS109 concepts)

| Module | Concept | Source |
| --- | --- | --- |
| Acceptance model | Logistic regression by MLE | 344,798 primary offers |
| Future kidney quality | Beta distribution fit by MLE | 1,019 donors |
| Offer arrival / survival | Exponential processes | SRTR-published rates |
| The futures | Monte Carlo simulation | 8,000 paths per decision |
| Which choice wins | Bayesian posterior + 95% CI | Beta(1,1) prior, bootstrap CI |
| What matters most | Shannon entropy / information gain | per-variable randomization |
| Accept vs. wait | Thompson sampling | Beta arms over outcomes |

## Run it

```bash
pip install -r requirements.txt
python fit_models.py          # fits all distributions -> models/params.json
uvicorn app:app --reload      # serves the experience at http://localhost:8000
```

The data files (`data/*.pkl`) are private UNOS data and are not included in this repo.
With them in `data/`, `fit_models.py` regenerates `models/params.json`. The frontend also
ships with a built-in mock so the experience runs even without the backend.

## Files

```
fit_models.py        learns every distribution from the data
simulate.py          the Monte Carlo + Bayesian + info-theory + Thompson engine
app.py               FastAPI server
templates/index.html the full five-screen experience
models/params.json   fitted parameters (generated)
static/*.png         screenshots of each screen
```

## A note on the data

This uses a randomized/synthetic release of UNOS kidney offer data and SRTR-published
population rates. It is an educational probability experience, **not** a clinical tool, and is
not intended to inform real transplant decisions.
