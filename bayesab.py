import joblib
import numpy as np
import pandas as pd
import scipy.stats as stats


class Experiment():
    def __init__(self, model_A, test_A, model_B, test_B, labels):
        self.model_A = model_A
        self.test_A = test_A
        self.model_B = model_B
        self.test_B = test_B
        self.labels = labels
        self.eval_A = self.labels.eq(self.model_A.predict(self.test_A))
        self.eval_B = self.labels.eq(self.model_B.predict(self.test_B))

    def proba_B_gt_A(self) -> float:
        SIM_SIZE = 1_000_000
        prior_alpha_A, prior_beta_A, prior_alpha_B, prior_beta_B = 1, 1, 1, 1

        self.alpha_A = prior_alpha_A + self.eval_A.sum()
        self.beta_A = prior_beta_A + (len(self.eval_A) - self.alpha_A)
        self.posterior_A = stats.beta(a=self.alpha_A, b=self.beta_A)
        sim_A = self.posterior_A.rvs(size=SIM_SIZE)

        self.alpha_B = prior_alpha_B + self.eval_B.sum()
        self.beta_B = prior_beta_B + (len(self.eval_B) - self.alpha_B)
        self.posterior_B = stats.beta(a=self.alpha_B, b=self.beta_B)
        sim_B = self.posterior_B.rvs(size=SIM_SIZE)

        return (sim_B > sim_A).sum() / SIM_SIZE

    def make_chart(self, **kwargs):
        p_B_gt_A = self.proba_B_gt_A()
        params = np.linspace(0, 1, 10001)
        model_A_curve = pd.Series(self.posterior_A.pdf(params), index=params)
        model_B_curve = pd.Series(self.posterior_B.pdf(params), index=params)
        title = f"Model A vs. Model B\nP(B > A) = {round(p_B_gt_A, 3)}"
        minA, maxA = self.posterior_A.interval(.9999)
        minB, maxB = self.posterior_B.interval(.9999)
        xlim = (min(minA, minB), max(maxA, maxB))
        ax = model_A_curve.plot(label="Model A", legend=True, title=title, xlim=xlim)
        model_B_curve.plot(label="Model B", legend=True, ax=ax, xlabel="Accuracy", ylabel="Likelihood", **kwargs)

    def ab_time_estimate(self, confidence=0.95, quantile=0.05) -> pd.Timedelta:
        SIM_SIZE = 10_000
        eval_sim_A = [self.simulate_eval_data(self.eval_A) for _ in range(SIM_SIZE)]
        eval_sim_B = [self.simulate_eval_data(self.eval_B) for _ in range(SIM_SIZE)]
        self.p_curves = joblib.Parallel(n_jobs=-2, verbose=3)(joblib.delayed(self.make_p_curve)(a, b) for a, b in zip(eval_sim_A, eval_sim_B))
        self.p_quantiles = pd.DataFrame(self.p_curves).quantile(quantile)
        if not self.p_quantiles.ge(confidence).any():
            estimate = pd.Timedelta(999, 'd')
        else:
            completion = self.p_quantiles.ge(confidence).idxmax()
            estimate = completion - self.p_quantiles.index.min()
        return estimate

    def simulate_eval_data(self, eval_data: pd.Series) -> pd.Series:
        counts = eval_data.resample('d').size()
        a = eval_data.sum()
        b = len(eval_data) - a
        posterior_predictive = stats.betabinom(n=1, a=a, b=b)
        return counts.apply(posterior_predictive.rvs).explode()

    def make_beta_params(self, data):
        alpha = data.sum()
        beta = len(data) - alpha
        return {"a": alpha + 1, "b": beta + 1}

    def calc_proba(self, data):
        return (data["B"] > data["A"]).sum() / len(data)

    def make_p_curve(self, eval_A, eval_B) -> dict[pd.Timestamp, np.float64]:
        SIM_SIZE = 10_000
        eval_A_alpha, eval_A_beta, eval_B_alpha, eval_B_beta = 1, 1, 1, 1
        eval_days = eval_A.index.union(eval_B.index).unique().sort_values()
        p_curve = {}

        for day in eval_days:
            data_A = eval_A.loc[day]
            eval_A_alpha += data_A.sum()
            eval_A_beta += len(data_A) - data_A.sum()
            rvs_A = stats.beta.rvs(eval_A_alpha, eval_A_beta, size=SIM_SIZE)

            data_B = eval_B.loc[day]
            eval_B_alpha += data_B.sum()
            eval_B_beta += len(data_B) - data_B.sum()
            rvs_B = stats.beta.rvs(eval_B_alpha, eval_B_beta, size=SIM_SIZE)

            p_curve[day] = (rvs_B > rvs_A).sum() / SIM_SIZE
        # eval_df = pd.DataFrame({'A': eval_A, 'B': eval_B})
        # param_df = eval_df.resample('d').apply(self.make_beta_params)
        # rvs_df = param_df.map(lambda x: stats.beta.rvs(size=SIM_SIZE, **x))
        # p_curve = rvs_df.apply(self.calc_proba, axis=1)
        return p_curve
