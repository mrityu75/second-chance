"""
Second Chance — Model Fitting
==============================
Learns every probability distribution the simulation needs from real UNOS
kidney offer data (April 2021), then exports them to a single JSON file that
the web experience loads.

Every model here maps directly to a CS109 concept:
  - Logistic regression (MLE)        -> P(accept | kidney, patient, queue position)
  - Beta distribution (MLE)          -> P(KDPI of a future kidney)
  - Exponential distribution         -> time until next offer / survival
  - Empirical conditional means      -> acceptance rates by feature bucket

Run:  python fit_models.py
Out:  models/params.json
"""

import pickle
import json
import numpy as np
import pandas as pd
from scipy import optimize, stats

RNG = np.random.default_rng(109)
DATA = "data"
OUT = "models/params.json"

# Published clinical constants (SRTR 2022 Annual Data Report + USRDS).
# We cite these in the writeup; they are not fit from our one-month sample.
DIALYSIS_ANNUAL_MORTALITY = 0.17          # ~17% of waitlisted dialysis patients die per year
LAMBDA_DEATH = DIALYSIS_ANNUAL_MORTALITY / 365.0   # daily death hazard, no transplant
MEDIAN_DAYS_BETWEEN_OFFERS = 30.0         # typical for a blood-type-O candidate
LAMBDA_ARRIVAL = 1.0 / MEDIAN_DAYS_BETWEEN_OFFERS
POST_TX_ANNUAL_MORTALITY_BASE = 0.04      # ~4%/yr baseline post-transplant, scaled by KDPI


def load():
    with open(f"{DATA}/offers.pkl", "rb") as f:
        offers = pickle.load(f)
    with open(f"{DATA}/donors.pkl", "rb") as f:
        donors = pickle.load(f)
    with open(f"{DATA}/patients.pkl", "rb") as f:
        patients = pickle.load(f)
    return offers, donors, patients


def build_decision_table(offers, donors, patients):
    """One row per primary offer with the features that drive accept/reject."""
    primary = offers[offers["INITIAL_RESPONSE"] == "Z"].copy()
    df = primary.merge(donors[["ID_1", "KDPI", "KDRI_RAO", "AGE_DON"]],
                       on="ID_1", how="left")
    df = df.merge(patients[["ID_2", "INIT_CPRA", "INIT_EPTS", "INIT_AGE"]],
                  on="ID_2", how="left")
    df["accepted"] = (df["OFFER_ACCEPT"] == "Y").astype(int)
    df["seq"] = df["PTR_SEQUENCE_NUM"].fillna(df["PTR_SEQUENCE_NUM"].median())
    df = df.dropna(subset=["KDPI", "INIT_CPRA", "INIT_EPTS"])
    return df


def fit_logistic(df):
    """
    Logistic regression by maximum likelihood (CS109 Part 5).
    P(accept=1 | x) = sigmoid(w . x + b).
    Features chosen for interpretability + signal we verified in the data:
      kdpi (kidney quality), log_seq (queue depth = how many said no first),
      cpra (sensitization), epts (patient urgency).
    We standardize features so the learned weights are directly comparable
    as 'importance', which the InfoLingo-style explanation screen uses.
    """
    feat_cols = ["KDPI", "log_seq", "INIT_CPRA_s", "INIT_EPTS"]
    d = df.copy()
    d["log_seq"] = np.log1p(d["seq"])
    d["INIT_CPRA_s"] = d["INIT_CPRA"] / 100.0

    X = d[feat_cols].to_numpy(dtype=float)
    y = d["accepted"].to_numpy(dtype=float)

    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    Xs = np.hstack([Xs, np.ones((Xs.shape[0], 1))])  # bias term

    def neg_log_lik(w):
        z = Xs @ w
        # stable log-likelihood
        ll = np.sum(y * z - np.logaddexp(0, z))
        # mild L2 so weights stay sane on a 0.36%-positive dataset
        ll -= 0.5 * 1e-3 * np.sum(w[:-1] ** 2)
        return -ll

    w0 = np.zeros(Xs.shape[1])
    res = optimize.minimize(neg_log_lik, w0, method="L-BFGS-B")
    w = res.x

    return {
        "features": feat_cols,
        "weights": w[:-1].tolist(),
        "bias": float(w[-1]),
        "feature_mean": mu.tolist(),
        "feature_std": sd.tolist(),
        "base_rate": float(y.mean()),
    }


def fit_kdpi_beta(donors):
    """
    Fit Beta(a, b) to the empirical KDPI distribution by MLE (CS109 Part 4/5).
    Beta is the natural model for a quantity in [0, 1]; we sample future
    kidneys from it in the Monte Carlo engine.
    """
    kdpi = donors["KDPI"].dropna().to_numpy()
    kdpi = np.clip(kdpi, 1e-4, 1 - 1e-4)
    a, b, loc, scale = stats.beta.fit(kdpi, floc=0, fscale=1)
    return {"alpha": float(a), "beta": float(b),
            "empirical_mean": float(kdpi.mean()),
            "empirical_std": float(kdpi.std()),
            "n_donors": int(len(kdpi))}


def acceptance_by_bucket(df):
    """Empirical P(accept) by feature bucket — used for the 'why' explanations."""
    out = {}

    kb = pd.cut(df["KDPI"], [0, .2, .4, .6, .8, 1.0])
    out["by_kdpi"] = {str(k): {"rate": float(v["accepted"]),
                               "n": int(v["count"])}
                      for k, v in df.groupby(kb, observed=True)["accepted"]
                      .agg(["mean", "count"]).rename(
                          columns={"mean": "accepted", "count": "count"}).iterrows()}

    sb = pd.cut(df["seq"], [0, 10, 50, 100, 500, 100000])
    out["by_seq"] = {str(k): {"rate": float(v["accepted"]),
                              "n": int(v["count"])}
                     for k, v in df.groupby(sb, observed=True)["accepted"]
                     .agg(["mean", "count"]).rename(
                         columns={"mean": "accepted", "count": "count"}).iterrows()}

    df["high_cpra"] = df["INIT_CPRA"] > 50
    cg = df.groupby("high_cpra")["accepted"].agg(["mean", "count"])
    out["by_cpra"] = {("high" if k else "low"):
                      {"rate": float(r["mean"]), "n": int(r["count"])}
                      for k, r in cg.iterrows()}
    return out


def fit_wait_time(patients):
    """Exponential model for time-on-waitlist from GTIME_KI (CS109 Part 2)."""
    g = patients["GTIME_KI"].dropna()
    g = g[g > 0].to_numpy()
    lam = 1.0 / g.mean()
    return {"lambda_per_day": float(lam),
            "mean_days": float(g.mean()),
            "median_days": float(np.median(g)),
            "n": int(len(g))}


def main():
    print("Loading data...")
    offers, donors, patients = load()

    print("Building decision table...")
    df = build_decision_table(offers, donors, patients)
    print(f"  {len(df):,} primary offers, {int(df['accepted'].sum()):,} accepts "
          f"({df['accepted'].mean()*100:.3f}%)")

    print("Fitting logistic acceptance model (MLE)...")
    logit = fit_logistic(df)
    print(f"  weights {logit['features']} = "
          f"{[round(w,3) for w in logit['weights']]}")

    print("Fitting KDPI Beta distribution (MLE)...")
    kdpi_beta = fit_kdpi_beta(donors)
    print(f"  Beta(a={kdpi_beta['alpha']:.2f}, b={kdpi_beta['beta']:.2f})")

    print("Fitting wait-time exponential...")
    wait = fit_wait_time(patients)
    print(f"  mean wait {wait['mean_days']:.0f} days")

    print("Computing empirical acceptance buckets...")
    buckets = acceptance_by_bucket(df)

    params = {
        "meta": {
            "n_offers": int(len(df)),
            "n_accepts": int(df["accepted"].sum()),
            "date_range": "2021-04-01 to 2021-04-30",
            "source": "UNOS kidney offer data (synthetic/randomized release)",
        },
        "acceptance_logit": logit,
        "kdpi_beta": kdpi_beta,
        "wait_time": wait,
        "buckets": buckets,
        "clinical_constants": {
            "lambda_death_per_day": LAMBDA_DEATH,
            "dialysis_annual_mortality": DIALYSIS_ANNUAL_MORTALITY,
            "lambda_arrival_per_day": LAMBDA_ARRIVAL,
            "median_days_between_offers": MEDIAN_DAYS_BETWEEN_OFFERS,
            "post_tx_annual_mortality_base": POST_TX_ANNUAL_MORTALITY_BASE,
        },
    }

    with open(OUT, "w") as f:
        json.dump(params, f, indent=2)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
