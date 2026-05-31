"""
Second Chance — Simulation Engine
=================================
The probabilistic heart. Given one patient and one kidney offer on the table,
it simulates thousands of possible futures under two decisions:

    ACCEPT  -> take this kidney now
    WAIT    -> reject, stay on the list, hope for a better offer

Each simulated future unfolds day by day as a sequence of random events and
ends in exactly one outcome: TRANSPLANTED, STILL_WAITING, or DIED_WAITING.

CS109 concepts, all load-bearing:
  - Exponential RV        : time until next offer, time until death
  - Bernoulli RV          : does the patient accept an arriving offer?
  - Beta RV               : quality (KDPI) of each future kidney
  - Monte Carlo           : the futures themselves
  - Bayesian inference    : posterior P(waiting is the better choice)
  - Information theory     : which variable explains the most outcome entropy
  - Thompson sampling      : accept vs wait as two competing arms
"""

import json
import math
import numpy as np

TRANSPLANTED, STILL_WAITING, DIED_WAITING = 0, 1, 2
OUTCOME_NAMES = ["Transplanted", "Still waiting", "Died waiting"]


class SecondChance:
    def __init__(self, params_path="models/params.json", seed=109):
        with open(params_path) as f:
            self.p = json.load(f)
        self.rng = np.random.default_rng(seed)

        c = self.p["clinical_constants"]
        self.lambda_death = c["lambda_death_per_day"]
        self.lambda_arrival = c["lambda_arrival_per_day"]
        self.post_tx_base = c["post_tx_annual_mortality_base"]

        kb = self.p["kdpi_beta"]
        self.kdpi_a, self.kdpi_b = kb["alpha"], kb["beta"]

        lg = self.p["acceptance_logit"]
        self.w = np.array(lg["weights"], dtype=float)
        self.bias = lg["bias"]
        self.fmu = np.array(lg["feature_mean"], dtype=float)
        self.fsd = np.array(lg["feature_std"], dtype=float)

    # ---- core probability pieces -------------------------------------------

    def p_accept(self, kdpi, seq, cpra, epts):
        """Logistic acceptance probability for ONE offer to THIS patient."""
        x = np.array([kdpi, math.log1p(seq), cpra / 100.0, epts], dtype=float)
        xs = (x - self.fmu) / self.fsd
        z = float(xs @ self.w) + self.bias
        return 1.0 / (1.0 + math.exp(-z))

    def sample_future_kidney_kdpi(self, n=1):
        """Draw KDPI of a future offered kidney from the fitted Beta."""
        return self.rng.beta(self.kdpi_a, self.kdpi_b, size=n)

    def post_tx_daily_hazard(self, kdpi):
        """Worse kidney (higher KDPI) -> higher post-transplant daily hazard."""
        annual = self.post_tx_base * (1.0 + 2.0 * kdpi)   # 4%..12% per year
        return annual / 365.0

    def expected_graft_days(self, kdpi):
        """Mean survival after a transplant of given quality (for survival math)."""
        return 1.0 / self.post_tx_daily_hazard(kdpi)

    # ---- the two universes --------------------------------------------------

    def simulate_accept(self, kidney_kdpi, horizon_days=3650):
        """
        Universe A: take this kidney now. Two competing exponential clocks:
          - graft survival ~ Exp(post-tx hazard, worse for high KDPI)
          - patient death from other causes ~ Exp(small baseline)
        If the graft outlives the horizon -> TRANSPLANTED (success).
        If the patient dies first -> DIED_WAITING (a bad accept).
        Returns (outcome, survival_days).
        """
        graft_haz = self.post_tx_daily_hazard(kidney_kdpi)
        # post-transplant patients still face a small baseline mortality
        base_haz = 0.25 * self.lambda_death
        t_graft = self.rng.exponential(1.0 / graft_haz)
        t_death = self.rng.exponential(1.0 / base_haz)
        t = min(t_graft, t_death, horizon_days)
        if t_death < t_graft and t_death < horizon_days:
            return DIED_WAITING, t
        return TRANSPLANTED, t

    def simulate_wait(self, patient, horizon_days=3650, max_offers=400):
        """
        Universe B: reject and wait. Day by day:
          - draw time to next offer ~ Exp(lambda_arrival)
          - draw time to death      ~ Exp(lambda_death)
          - if death comes first -> DIED_WAITING
          - else a kidney arrives; draw its KDPI ~ Beta; the patient accepts
            with p_accept(...). Queue position improves as they wait.
        Returns (outcome, days_elapsed, n_offers_seen, accepted_kdpi or None).
        """
        cpra, epts = patient["cpra"], patient["epts"]
        elapsed = 0.0
        seq = patient.get("seq", 200)  # current position; improves over time

        for i in range(max_offers):
            t_offer = self.rng.exponential(1.0 / self.lambda_arrival)
            t_death = self.rng.exponential(1.0 / self.lambda_death)

            if t_death < t_offer:
                elapsed += t_death
                if elapsed >= horizon_days:
                    return STILL_WAITING, horizon_days, i, None
                return DIED_WAITING, elapsed, i, None

            elapsed += t_offer
            if elapsed >= horizon_days:
                return STILL_WAITING, horizon_days, i, None

            kdpi = float(self.sample_future_kidney_kdpi(1)[0])
            seq = max(1, seq * 0.93)  # each month waited, move up the list
            if self.rng.random() < self.p_accept(kdpi, seq, cpra, epts):
                return TRANSPLANTED, elapsed, i + 1, kdpi

        return STILL_WAITING, horizon_days, max_offers, None

    # ---- Monte Carlo over many futures -------------------------------------

    def run(self, patient, offer_kdpi, n=10000, horizon_days=3650):
        """
        Simulate n futures for ACCEPT and n for WAIT. Returns a rich dict the
        front end turns into the 'future galaxy' and the probability landscape.
        """
        acc_outcomes = np.empty(n, dtype=int)
        acc_survival = np.empty(n, dtype=float)
        for k in range(n):
            o, s = self.simulate_accept(offer_kdpi, horizon_days)
            acc_outcomes[k], acc_survival[k] = o, s

        wait_outcomes = np.empty(n, dtype=int)
        wait_days = np.empty(n, dtype=float)
        wait_kdpi = []
        for k in range(n):
            o, d, _, kd = self.simulate_wait(patient, horizon_days)
            wait_outcomes[k], wait_days[k] = o, d
            if kd is not None:
                wait_kdpi.append(kd)

        # survival-years proxy:
        #   accept  -> graft survival time (already in days)
        #   wait    -> if transplanted, expected graft life from that kidney;
        #              if died, the days survived; if still waiting, horizon
        acc_years = acc_survival / 365.0
        wait_years = np.where(
            wait_outcomes == TRANSPLANTED,
            wait_days / 365.0 + 8.0,            # got a graft after waiting
            np.where(wait_outcomes == DIED_WAITING,
                     wait_days / 365.0,          # died before transplant
                     horizon_days / 365.0))      # still waiting at horizon

        return {
            "n": n,
            "offer_kdpi": offer_kdpi,
            "accept": self._summary(acc_outcomes, acc_years),
            "wait": self._summary(wait_outcomes, wait_years),
            "accept_samples": self._galaxy(acc_outcomes),
            "wait_samples": self._galaxy(wait_outcomes),
            "posterior_wait_better": self._posterior_wait_better(
                acc_years, wait_years),
            "expected_years": {
                "accept": float(acc_years.mean()),
                "wait": float(wait_years.mean()),
            },
        }

    # ---- summaries / inference ---------------------------------------------

    def _summary(self, outcomes, years):
        counts = np.bincount(outcomes, minlength=3)
        return {
            "transplanted": int(counts[TRANSPLANTED]),
            "still_waiting": int(counts[STILL_WAITING]),
            "died_waiting": int(counts[DIED_WAITING]),
            "mean_years": float(years.mean()),
            "p25_years": float(np.percentile(years, 25)),
            "p75_years": float(np.percentile(years, 75)),
        }

    def _galaxy(self, outcomes, cap=2000):
        """Down-sample outcomes for the dot 'galaxy' visual."""
        if len(outcomes) > cap:
            idx = self.rng.choice(len(outcomes), cap, replace=False)
            outcomes = outcomes[idx]
        return outcomes.tolist()

    def _posterior_wait_better(self, acc_years, wait_years):
        """
        Bayesian posterior that WAIT yields more life-years than ACCEPT.
        Prior: Beta(1,1) (uniform — we genuinely don't know).
        Each paired future is a Bernoulli trial 'did wait beat accept?'.
        Posterior is Beta(1 + wins, 1 + losses); we report mean + 95% CI.
        """
        wins = int(np.sum(wait_years > acc_years))
        losses = len(acc_years) - wins
        a, b = 1 + wins, 1 + losses
        mean = a / (a + b)
        lo, hi = self._beta_ci(a, b)
        return {"mean": float(mean), "ci_low": float(lo), "ci_high": float(hi),
                "wins": wins, "losses": losses}

    def _beta_ci(self, a, b, level=0.95):
        from scipy.stats import beta as betadist
        lo = betadist.ppf((1 - level) / 2, a, b)
        hi = betadist.ppf(1 - (1 - level) / 2, a, b)
        return lo, hi

    # ---- information theory: which variable matters most? ------------------

    def information_gain(self, patient, offer_kdpi, n=3000):
        """
        How much does each patient variable reduce uncertainty about the
        accept-vs-wait outcome? We compute H(decision outcome) overall, then
        H(outcome | variable held at its value) by re-simulating with that
        variable randomized, and report the reduction in bits.
        The variable that reduces entropy the most is the InfoLingo moment.
        """
        base = self.run(patient, offer_kdpi, n=n)
        base_H = self._outcome_entropy(base)

        gains = {}
        for var, draw in [
            ("CPRA", lambda: self.rng.uniform(0, 100)),
            ("EPTS", lambda: self.rng.uniform(0, 1)),
            ("Queue position", lambda: self.rng.uniform(1, 500)),
        ]:
            randomized_H = []
            for _ in range(12):
                pp = dict(patient)
                if var == "CPRA":
                    pp["cpra"] = draw()
                elif var == "EPTS":
                    pp["epts"] = draw()
                else:
                    pp["seq"] = draw()
                r = self.run(pp, offer_kdpi, n=max(400, n // 8))
                randomized_H.append(self._outcome_entropy(r))
            gains[var] = max(0.0, base_H - float(np.mean(randomized_H)))

        total = sum(gains.values()) or 1.0
        return {"base_entropy_bits": base_H,
                "gain_bits": gains,
                "share": {k: v / total for k, v in gains.items()}}

    def _outcome_entropy(self, result):
        """Shannon entropy (bits) of the WAIT outcome distribution."""
        w = result["wait"]
        n = result["n"]
        ps = [w["transplanted"] / n, w["still_waiting"] / n, w["died_waiting"] / n]
        return -sum(p * math.log2(p) for p in ps if p > 0)

    # ---- Thompson sampling: accept vs wait as competing arms ----------------

    def thompson_recommendation(self, result):
        """
        Treat ACCEPT and WAIT as two arms whose 'reward' is a good outcome
        (transplanted and alive). Model each arm's success rate with a Beta
        posterior built from the simulation, then Thompson-sample to pick.
        """
        def arm(summary):
            good = summary["transplanted"]
            bad = summary["still_waiting"] + summary["died_waiting"]
            return 1 + good, 1 + bad

        a_acc, b_acc = arm(result["accept"])
        a_wait, b_wait = arm(result["wait"])

        draws = 5000
        s_acc = self.rng.beta(a_acc, b_acc, draws)
        s_wait = self.rng.beta(a_wait, b_wait, draws)
        wait_wins = float(np.mean(s_wait > s_acc))
        return {
            "recommend": "wait" if wait_wins > 0.5 else "accept",
            "p_wait_is_better": wait_wins,
            "accept_arm": {"alpha": a_acc, "beta": b_acc},
            "wait_arm": {"alpha": a_wait, "beta": b_wait},
        }


if __name__ == "__main__":
    sc = SecondChance()
    patient = {"cpra": 97, "epts": 0.62, "seq": 180}
    offer_kdpi = 0.43
    print("Simulating 10,000 futures for each decision...")
    res = sc.run(patient, offer_kdpi, n=10000)
    print(f"\nOffer KDPI: {offer_kdpi}")
    print(f"ACCEPT: {res['accept']}")
    print(f"WAIT:   {res['wait']}")
    print(f"Expected years  accept={res['expected_years']['accept']:.1f}  "
          f"wait={res['expected_years']['wait']:.1f}")
    post = res["posterior_wait_better"]
    print(f"P(waiting is better) = {post['mean']*100:.1f}% "
          f"(95% CI {post['ci_low']*100:.1f}-{post['ci_high']*100:.1f}%)")
    print("\nInformation gain:")
    ig = sc.information_gain(patient, offer_kdpi)
    for k, v in ig["share"].items():
        print(f"  {k}: {v*100:.0f}% of explained uncertainty")
    print("\nThompson recommendation:")
    print(" ", sc.thompson_recommendation(res))
