# """Core simulation library: DC motor, PI/PID (with saturation + anti-windup) and constrained MPC.

# State vector x = [ia, omega, theta]; inputs: armature voltage va, load torque Td (disturbance).
#     dia/dt    = (-Ra*ia - Kb*omega + va) / La
#     domega/dt = ( Kt*ia - B*omega - Td) / J
#     dtheta/dt = omega
# The plant is simulated with an exact zero-order-hold discretisation at sample time Ts.
# """
import time as _time

import warnings

import cvxpy as cp
import numpy as np
from scipy.signal import cont2discrete

DEFAULTS = dict(
    # motor
    J=0.01, B=0.1, Kt=0.1, Kb=0.1, Ra=1.0, La=0.5,
    # scenario
    ref1=10.0, ref2=6.0, t_step=3.0, Td=0.2, t_dist=5.5, T=8.0, Ts=0.02,
    # constraints
    Vmax=24.0, Imax=10.5, Wmax=20.0, use_I=True, use_W=True,
    # MPC
    Np=20, Nc=5, Q=1.0, R=0.05, rho=300.0, preview=True, Ld=0.7,
    # PI / PID
    kp_pi=2.0, ki_pi=5.0, kp_pid=2.0, ki_pid=5.0, kd_pid=0.02,
)
BIG = 1e6
warnings.filterwarnings("ignore", message="Solution may be inaccurate")


# ------------------------------------------------------------------ plant
def plant(P):
    A = np.array([[-P["Ra"] / P["La"], -P["Kb"] / P["La"], 0],
                  [P["Kt"] / P["J"], -P["B"] / P["J"], 0],
                  [0, 1, 0]])
    Bc = np.array([[1 / P["La"], 0], [0, -1 / P["J"]], [0, 0]])
    Ad, Bd, *_ = cont2discrete((A, Bc, np.eye(3), np.zeros((3, 2))), P["Ts"], method="zoh")
    return A, Ad, Bd


def ref_at(P, t):
    return P["ref1"] if t < P["t_step"] else P["ref2"]


def load_at(P, t):
    return P["Td"] if t >= P["t_dist"] else 0.0


def _loop(P, controller):
    """Closed loop: controller(k, t, x, u_prev) -> va."""
    _, Ad, Bd = plant(P)
    n = int(round(P["T"] / P["Ts"])) + 1
    t = np.arange(n) * P["Ts"]
    X, U, R, D, DH = np.zeros((n, 3)), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    x, u_prev = np.zeros(3), 0.0
    solve_t = []
    for k in range(n):
        t0 = _time.perf_counter()
        u = float(np.clip(controller(k, t[k], x, u_prev), -P["Vmax"], P["Vmax"]))
        solve_t.append(_time.perf_counter() - t0)
        X[k], U[k], R[k], D[k] = x, u, ref_at(P, t[k]), load_at(P, t[k])
        x = Ad @ x + Bd @ np.array([u, D[k]])
        u_prev = u
    return dict(t=t, ia=X[:, 0], speed=X[:, 1], pos=X[:, 2], v=U, ref=R, load=D,
                solve_ms=1e3 * float(np.mean(solve_t)))


# ------------------------------------------------------------------ PI / PID
def run_pid(P, kind="PID"):
    kp = P["kp_pi"] if kind == "PI" else P["kp_pid"]
    ki = P["ki_pi"] if kind == "PI" else P["ki_pid"]
    kd = 0.0 if kind == "PI" else P["kd_pid"]
    Ts, V = P["Ts"], P["Vmax"]
    st = dict(I=0.0, e_prev=0.0, d_f=0.0)

    def ctrl(k, t, x, u_prev):
        e = ref_at(P, t) - x[1]
        d_raw = (e - st["e_prev"]) / Ts if k > 0 else 0.0
        st["d_f"] = 0.8 * st["d_f"] + 0.2 * d_raw          # filtered derivative
        I_new = st["I"] + ki * e * Ts
        u = kp * e + I_new + kd * st["d_f"]
        if abs(u) > V and e * u > 0:                       # clamping anti-windup
            I_new = st["I"]
            u = kp * e + I_new + kd * st["d_f"]
        st["I"], st["e_prev"] = I_new, e
        return u

    out = _loop(P, ctrl)
    out["dhat"] = np.zeros_like(out["t"])
    return out


# ------------------------------------------------------------------ MPC
class MPC:
    """Offset-free input-increment MPC (QP) with soft state constraints.

    min  sum_{k=1..Np} Q (w_k - r_k)^2 + sum_{k=0..Nc-1} R du_k^2 + rho * sum (s_i^2 + s_w^2)
    s.t. x_{k+1} = Ad x_k + Bu u_k + Ed d_hat,  u_k = u_{k-1} + du_k (frozen after Nc moves)
         |u_k| <= Vmax (hard),  |ia_k| <= Imax + s_i,  |w_k| <= Wmax + s_w,  s >= 0
    """

    def __init__(self, P, Ad, Bd):
        Np, Nc = int(P["Np"]), min(int(P["Nc"]), int(P["Np"]))
        self.Np, self.P = Np, P
        Bu, Ed = Bd[:, 0], Bd[:, 1]
        self.Ed = Ed
        Imax = P["Imax"] if P["use_I"] else BIG
        Wmax = P["Wmax"] if P["use_W"] else BIG
        self.x0, self.up = cp.Parameter(3), cp.Parameter()
        self.d, self.r = cp.Parameter(), cp.Parameter(Np)
        du = cp.Variable(Nc)
        X = cp.Variable((3, Np + 1))
        si, sw = cp.Variable(Np, nonneg=True), cp.Variable(Np, nonneg=True)
        T = np.zeros((Np, Nc))
        for k in range(Np):
            T[k, :min(k + 1, Nc)] = 1.0
        u = self.up * np.ones(Np) + T @ du
        cons = [X[:, 0] == self.x0,
                X[:, 1:] == Ad @ X[:, :-1] + Bu.reshape(3, 1) @ cp.reshape(u, (1, Np), order="F")
                + np.outer(Ed, np.ones(Np)) * self.d,
                u <= P["Vmax"], u >= -P["Vmax"],
                X[0, 1:] <= Imax + si, X[0, 1:] >= -Imax - si,
                X[1, 1:] <= Wmax + sw, X[1, 1:] >= -Wmax - sw]
        cost = (P["Q"] * cp.sum_squares(X[1, 1:] - self.r) + P["R"] * cp.sum_squares(du)
                + P["rho"] * (cp.sum_squares(si) + cp.sum_squares(sw)))
        self.prob, self.du = cp.Problem(cp.Minimize(cost), cons), du

    def solve(self, x, u_prev, dhat, rseq):
        self.x0.value, self.up.value, self.d.value, self.r.value = x, u_prev, dhat, rseq
        try:
            self.prob.solve(solver=cp.OSQP, warm_start=True, eps_abs=1e-5, eps_rel=1e-5, max_iter=20000)
            if self.du.value is not None:
                return u_prev + float(self.du.value[0])
        except cp.SolverError:
            pass
        return u_prev


def run_mpc(P):
    _, Ad, Bd = plant(P)
    mpc = MPC(P, Ad, Bd)
    Bu, Ed, Ts = Bd[:, 0], Bd[:, 1], P["Ts"]
    st = dict(x=None, u=0.0, dh=0.0)
    dh_log = []

    def ctrl(k, t, x, u_prev):
        if st["x"] is not None:                            # disturbance estimator (innovation on speed)
            pred = (Ad @ st["x"] + Bu * st["u"] + Ed * st["dh"])[1]
            st["dh"] += P["Ld"] * (x[1] - pred) / Ed[1]
        rseq = np.array([ref_at(P, t + (j + 1) * Ts if P["preview"] else t) for j in range(mpc.Np)])
        u = mpc.solve(x, u_prev, st["dh"], rseq)
        st["x"], st["u"] = x.copy(), float(np.clip(u, -P["Vmax"], P["Vmax"]))
        dh_log.append(st["dh"])
        return st["u"]

    out = _loop(P, ctrl)
    out["dhat"] = np.array(dh_log)
    return out


# ------------------------------------------------------------------ metrics
def _step_metrics(t, y, t0, t1, y0, yf):
    m = (t >= t0) & (t <= t1)
    tt, yy = t[m] - t0, y[m]
    dlt = yf - y0
    s = np.sign(dlt)
    lo, hi = y0 + 0.1 * dlt, y0 + 0.9 * dlt
    i10 = np.argmax(s * (yy - lo) >= 0) if np.any(s * (yy - lo) >= 0) else None
    i90 = np.argmax(s * (yy - hi) >= 0) if np.any(s * (yy - hi) >= 0) else None
    rise = tt[i90] - tt[i10] if i10 is not None and i90 is not None else np.nan
    out_band = np.where(np.abs(yy - yf) > 0.02 * abs(dlt))[0]
    settle = (tt[out_band[-1]] + (tt[1] - tt[0]) if len(out_band) else 0.0)
    if len(out_band) and out_band[-1] == len(tt) - 1:
        settle = np.nan
    os_ = max(0.0, s * (yy.max() if s > 0 else yy.min()) - s * yf) / abs(dlt) * 100
    tail = yy[-max(3, len(yy) // 10):]
    return rise, settle, os_, yf - float(np.mean(tail))


def compute_metrics(P, r):
    t, w = r["t"], r["speed"]
    t_end = P["T"]
    la = P["Np"] * P["Ts"] if (r.get("kind") == "MPC" and P["preview"]) else 0.0   # exclude pre-emptive preview window
    t_d = P["t_dist"] if P["Td"] != 0 else t_end
    seg1_end = min(P["t_step"], t_d)
    rows = {}
    a = _step_metrics(t, w, 0.0, seg1_end - la - P["Ts"], 0.0, P["ref1"])
    rows["Rise time, 1st step (s)"], rows["Settling time 2%, 1st step (s)"] = a[0], a[1]
    rows["Overshoot, 1st step (%)"], rows["Steady-state error, 1st step"] = a[2], a[3]
    if P["t_step"] < t_end and P["ref2"] != P["ref1"]:
        end2 = min(t_d, t_end) if t_d > P["t_step"] else t_end
        b = _step_metrics(t, w, P["t_step"], end2 - la - P["Ts"], P["ref1"], P["ref2"])
        rows["Settling time 2%, ref change (s)"], rows["Overshoot, ref change (%)"] = b[1], b[2]
    if P["Td"] != 0 and P["t_dist"] < t_end:
        m = t >= P["t_dist"]
        dev = w[m] - ref_at(P, t_end)
        rows["Disturbance: peak speed dip (rad/s)"] = float(np.max(np.abs(dev)))
        out = np.where(np.abs(dev) > 0.02 * abs(ref_at(P, t_end)))[0]
        rows["Disturbance: recovery time 2% (s)"] = (t[m][out[-1]] + P["Ts"] - P["t_dist"]) if len(out) else 0.0
        rows["Steady-state error, after disturbance"] = float(ref_at(P, t_end) - np.mean(w[-max(3, len(w) // 25):]))
    rows["IAE (rad)"] = float(np.sum(np.abs(r["ref"] - w)) * P["Ts"])
    rows["Peak |voltage| (V)"] = float(np.max(np.abs(r["v"])))
    rows["Peak |current| (A)"] = float(np.max(np.abs(r["ia"])))
    rows["Peak speed (rad/s)"] = float(np.max(np.abs(w)))
    tol = 1e-3
    rows["Voltage limit violated"] = "No" if rows["Peak |voltage| (V)"] <= P["Vmax"] + tol else "YES"
    over_i = np.maximum(0, np.abs(r["ia"]) - P["Imax"])
    rows["Current violation: max (A) / duration (s)"] = f"{over_i.max():.2f} / {np.sum(over_i > 0.01) * P['Ts']:.2f}"
    over_w = np.maximum(0, np.abs(w) - P["Wmax"])
    rows["Speed violation: max (rad/s)"] = f"{over_w.max():.3f}"
    rows["Mean compute time / step (ms)"] = r["solve_ms"]
    out = {}
    for k, v in rows.items():
        out[k] = round(float(v), 4) if isinstance(v, (float, np.floating)) else v
    return out


def run(P, kind):
    r = run_mpc(P) if kind == "MPC" else run_pid(P, kind)
    r["kind"] = kind
    return r