"""
Generate a replica of the AI4I 2020 Predictive Maintenance dataset.

Only used when the real file (data/ai4i2020.csv, from UCI or Kaggle) is not present.
It follows the generation process published with the dataset (Matzka, 2020):
  - Product quality Type L/M/H in 50/30/20 proportion
  - Air temperature ~300 K (random walk, sd 2 K); process temp = air + 10 K (sd 1 K)
  - Torque ~ N(40, 10) Nm; rotational speed negatively correlated with torque (~1538 rpm mean)
  - Tool wear accumulates 5/3/2 min per process for H/M/L, tool replaced ~200-240 min
  - Failure modes:
      TWF tool wear failure  : random failure when tool wear is 200-240 min
      HDF heat dissipation   : (process - air temp) < 8.6 K AND speed < 1380 rpm
      PWF power failure      : power < 3500 W or > 9000 W
      OSF overstrain failure : tool wear x torque > 11000 (L) / 12000 (M) / 13000 (H) min.Nm
      RNF random failure     : 0.1% of processes
Columns match the real file exactly, so the pipeline is identical for both.
"""
import numpy as np
import pandas as pd


def generate(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    types = rng.choice(["L", "M", "H"], size=n, p=[0.5, 0.3, 0.2])

    # Temperatures as smooth random walks (shift-level drift)
    def walk(sd, phi=0.97):
        # mean-reverting random walk (AR(1)), normalised to the target sd
        e = rng.normal(0, 1, n)
        w = np.zeros(n)
        for i in range(1, n):
            w[i] = phi * w[i - 1] + e[i]
        w = (w - w.mean()) / w.std()
        return w * sd

    air = 300 + walk(2.0)
    proc = air + 10 + walk(1.0)

    torque = np.clip(rng.normal(40, 9.3, n), 3.8, None)
    speed = 1538 - 15.8 * (torque - 40) + rng.normal(0, 70, n)
    speed = np.clip(speed, 1168, 2886)

    # Tool wear: increments by type, tool replaced at random point in 200-240 min
    inc = np.where(types == "H", 5, np.where(types == "M", 3, 2))
    wear = np.zeros(n)
    w = 0.0
    limit = rng.uniform(200, 240)
    twf = np.zeros(n, dtype=int)
    for i in range(n):
        w += inc[i]
        if w >= limit:
            # tool either fails (~43%) or is replaced at this point
            if rng.random() < 0.43:
                twf[i] = 1
            wear[i] = w
            w = 0.0
            limit = rng.uniform(200, 240)
            continue
        wear[i] = w
    wear = np.round(wear).astype(int)

    omega = speed * 2 * np.pi / 60
    power = torque * omega
    hdf = ((proc - air) < 8.6) & (speed < 1380)
    pwf = (power < 3500) | (power > 9000)
    osf_lim = np.where(types == "L", 11000, np.where(types == "M", 12000, 13000))
    osf = (wear * torque) > osf_lim
    rnf = rng.random(n) < 0.001

    df = pd.DataFrame({
        "UDI": np.arange(1, n + 1),
        "Product ID": [f"{t}{rng.integers(10000, 99999)}" for t in types],
        "Type": types,
        "Air temperature [K]": np.round(air, 1),
        "Process temperature [K]": np.round(proc, 1),
        "Rotational speed [rpm]": np.round(speed).astype(int),
        "Torque [Nm]": np.round(torque, 1),
        "Tool wear [min]": wear,
        "TWF": twf,
        "HDF": hdf.astype(int),
        "PWF": pwf.astype(int),
        "OSF": osf.astype(int),
        "RNF": rnf.astype(int),
    })
    df.insert(8, "Machine failure",
              (df[["TWF", "HDF", "PWF", "OSF", "RNF"]].sum(axis=1) > 0).astype(int))
    return df


if __name__ == "__main__":
    d = generate()
    d.to_csv("data/ai4i2020_synthetic.csv", index=False)
    print(d.describe().T[["mean", "std", "min", "max"]])
    print(d[["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]].sum())
