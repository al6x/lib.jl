import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.special import kv
from scipy.optimize import minimize

# ---------- Student-t ----------
def t_pdf(x, nu, s):
  a = math.gamma((nu + 1)/2)/(math.gamma(nu/2)*math.sqrt(math.pi*nu)*s)
  z = 1.0 + (x/s)**2/nu
  return a * z**(-(nu + 1)/2)

# ---------- Loss (same as before) ----------
def simple_weighted_loss(x, f_ref, nu, s):
  fr = np.maximum(f_ref, 1e-300)
  ft = np.maximum(t_pdf(x, nu, s), 1e-300)
  w = np.sqrt(fr)
  return np.sum(w * (np.log(fr) - np.log(ft))**2)

# ---------- 2-parameter optimizer (same loss) ----------
def fit_student_t_simple(x, f_ref, nu0=6.0):
  x = np.asarray(x, float); f_ref = np.asarray(f_ref, float)
  w = f_ref / f_ref.sum()
  mu = np.sum(x * w)
  var = np.sum((x - mu)**2 * w)
  std = math.sqrt(max(var, 1e-12))
  s0 = std if nu0 <= 2 else std * math.sqrt((nu0 - 2.0)/nu0)
  obj = lambda p: simple_weighted_loss(x, f_ref, p[0], p[1])
  res = minimize(obj, x0=[nu0, s0], method="L-BFGS-B",
                 bounds=[(1e-3, None), (1e-12, None)])
  return float(res.x[0]), float(res.x[1])

# ---------- Symmetric NIG ----------
def nig_pdf(x, alpha, delta, mu=0.0):
  x = np.asarray(x, dtype=float)
  r = np.sqrt(delta**2 + (x - mu)**2)
  z = np.maximum(alpha * r, 1e-300)
  K1 = kv(1.0, z)
  num = alpha * delta * np.exp(delta * alpha) * K1
  den = math.pi * r
  return num / den

# ---------- CDF from pdf on grid + adaptive x ----------
def cdf_from_pdf_grid(x, f):
  F = np.cumsum((f[1:] + f[:-1]) * np.diff(x) * 0.5)
  F = np.concatenate(([0.0], F))
  F /= F[-1]
  return F

def adaptive_grid_from_pdf(pdf_func, x_span=(-12,12), tq=1e-4, n_eqprob=60, n_equid=60):
  xg = np.linspace(x_span[0], x_span[1], 4000)
  fg = pdf_func(xg)
  Fg = cdf_from_pdf_grid(xg, fg)
  qs = np.linspace(tq, 1 - tq, n_eqprob)
  x_prob = np.interp(qs, Fg, xg)
  x_lo, x_hi = float(x_prob[0]), float(x_prob[-1])
  x_quan = np.linspace(x_lo, x_hi, n_equid)
  x = np.unique(np.sort(np.concatenate([x_prob, x_quan])))
  return x, x_lo, x_hi

# ---------- Cases: symmetric NIG (beta=0) ----------
# alpha controls exponential tail rate; delta is scale. Ranges chosen for daily→annual.
nig_cases = [
  # Daily-scale: heavy tails, sharp head
  {"name":"Daily — NIG α=0.7, δ=1.0", "alpha":0.7, "delta":1.0},
  {"name":"Daily — NIG α=1.2, δ=1.0", "alpha":1.2, "delta":1.0},

  # Monthly: thinner tails
  {"name":"Monthly — NIG α=3.0, δ=1.0", "alpha":3.0, "delta":1.0},
  {"name":"Monthly — NIG α=5.0, δ=1.0", "alpha":5.0, "delta":1.0},

  # Annual / near-Gaussian
  {"name":"Annual — NIG α=8.0, δ=1.0", "alpha":8.0, "delta":1.0},
  {"name":"Annual — NIG α=10.0, δ=0.5", "alpha":10.0, "delta":0.5},
]

# ---------- Iterate, fit Student-t, and plot each separately ----------
for case in nig_cases:
  alpha, delta = case["alpha"], case["delta"]
  pdf_ref = lambda x: nig_pdf(x, alpha=alpha, delta=delta, mu=0.0)

  # grid
  x, x_lo, x_hi = adaptive_grid_from_pdf(pdf_ref, x_span=(-12,12), tq=1e-4, n_eqprob=60, n_equid=60)

  # reference pdf (NIG)
  f_ref = pdf_ref(x)

  # fit Student-t
  nu_hat, s_hat = fit_student_t_simple(x, f_ref)
  f_t = t_pdf(x, nu_hat, s_hat)

  # linear plot
  plt.figure(figsize=(7,4))
  plt.plot(x, f_ref, 'o-', ms=3, label=case["name"])
  plt.plot(x, f_t,  'o--', ms=3, label=f"Student-t (ν={nu_hat:.2f}, s={s_hat:.3f})")
  plt.title(f"{case['name']} vs Student-t — linear")
  plt.xlabel("x"); plt.ylabel("pdf")
  plt.legend(); plt.tight_layout(); plt.show()

  # symlog x, log y plot
  plt.figure(figsize=(7,4))
  plt.plot(x, f_ref, 'o-', ms=3, label="NIG")
  plt.plot(x, f_t,  'o--', ms=3, label="Student-t")
  plt.xscale('symlog', linthresh=1.0)
  plt.yscale('log')
  plt.title(f"{case['name']} vs Student-t — symlog x, log y")
  plt.xlabel("x"); plt.ylabel("pdf (log)")
  plt.legend(); plt.tight_layout(); plt.show()

  print(f"{case['name']}: x range [{x_lo:.5f}, {x_hi:.5f}], fitted t ν={nu_hat:.4f}, s={s_hat:.6f}")
