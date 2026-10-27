import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from motor_mpc import DEFAULTS, compute_metrics, run

st.set_page_config(page_title="DC Motor MPC vs PI/PID", layout="wide")
st.title("⚡ DC Motor Speed Control: Constrained MPC vs PI / PID")
st.caption("Exact ZOH-discretised DC motor, QP-based MPC (cvxpy/OSQP), anti-windup PI/PID baselines.")

D = DEFAULTS
sb = st.sidebar

sb.header("1. Controllers to compare")
sel = sb.multiselect("Controllers", ["PI", "PID", "MPC"], default=["PI", "PID", "MPC"])

sb.header("2. Motor parameters")
J = sb.number_input("Inertia J (kg·m²)", value=D["J"], format="%.4f")
B = sb.number_input("Viscous friction B (N·m·s)", value=D["B"], format="%.4f")
Kt = sb.number_input("Torque constant Kt", value=D["Kt"], format="%.4f")
Kb = sb.number_input("Back-EMF constant Kb", value=D["Kb"], format="%.4f")
Ra = sb.number_input("Resistance Ra (Ω)", value=D["Ra"], format="%.2f")
La = sb.slider("Inductance La (H)", 0.01, 2.0, D["La"], 0.01)

sb.header("3. Scenario")
ref1 = sb.number_input("Target speed (rad/s)", value=D["ref1"], step=1.0)
ref2 = sb.number_input("Speed after reference change (rad/s)", value=D["ref2"], step=1.0)
t_step = sb.slider("Reference change time (s)", 0.5, 9.0, D["t_step"], 0.1)
Td = sb.number_input("Load-torque disturbance (N·m)", value=D["Td"], step=0.05, format="%.2f")
t_dist = sb.slider("Disturbance time (s)", 0.5, 9.0, D["t_dist"], 0.1)
T = sb.slider("Simulation time (s)", 2.0, 10.0, D["T"], 0.5)
Ts = sb.select_slider("Sample time Ts (s)", [0.005, 0.01, 0.02, 0.03, 0.05], value=D["Ts"])

sb.header("4. Constraints")
Vmax = sb.number_input("Voltage limit ±Vmax (V)", value=D["Vmax"], step=1.0)
use_I = sb.checkbox("Enforce current limit (MPC)", value=True)
Imax = sb.number_input("Current limit ±Imax (A)", value=D["Imax"], step=1.0)
use_W = sb.checkbox("Enforce speed limit (MPC)", value=True)
Wmax = sb.number_input("Speed limit ±ωmax (rad/s)", value=D["Wmax"], step=1.0)

sb.header("5. MPC settings")
Np = sb.slider("Prediction steps Np", 2, 80, D["Np"])
Nc = sb.slider("Control adjustment steps Nc", 1, 40, D["Nc"])
Q = sb.number_input("Priority: speed tracking Q", value=D["Q"], format="%.3f")
R = sb.number_input("Priority: control effort R (on Δv)", value=D["R"], format="%.4f", step=0.005)
rho = sb.number_input("Priority: constraint softness ρ", value=D["rho"], format="%.1f")
preview = sb.checkbox("Reference preview over horizon", value=D["preview"])

sb.header("6. PI / PID gains")
kp_pi = sb.slider("PI  Kp", 0.0, 10.0, D["kp_pi"], 0.1)
ki_pi = sb.slider("PI  Ki", 0.0, 50.0, D["ki_pi"], 0.5)
kp_pid = sb.slider("PID Kp", 0.0, 10.0, D["kp_pid"], 0.1)
ki_pid = sb.slider("PID Ki", 0.0, 50.0, D["ki_pid"], 0.5)
kd_pid = sb.slider("PID Kd", 0.0, 1.0, D["kd_pid"], 0.01)

P = dict(J=J, B=B, Kt=Kt, Kb=Kb, Ra=Ra, La=La, ref1=ref1, ref2=ref2, t_step=t_step, Td=Td,
         t_dist=t_dist, T=T, Ts=Ts, Vmax=Vmax, Imax=Imax, Wmax=Wmax, use_I=use_I, use_W=use_W,
         Np=Np, Nc=min(Nc, Np), Q=Q, R=R, rho=rho, preview=preview, Ld=D["Ld"],
         kp_pi=kp_pi, ki_pi=ki_pi, kp_pid=kp_pid, ki_pid=ki_pid, kd_pid=kd_pid)
if Nc > Np:
    sb.warning("Nc cannot exceed Np; using Nc = Np.")


@st.cache_data(show_spinner="Simulating…")
def simulate(P, kind):
    r = run(P, kind)
    return r, compute_metrics(P, r)


COL = {"PI": "tab:blue", "PID": "tab:green", "MPC": "tab:orange"}
tabs = st.tabs(["📈 Responses", "📋 Performance metrics", "🔬 MPC parameter study", "📐 Formulation"])

if not sel:
    st.warning("Select at least one controller in the sidebar.")
    st.stop()
res = {k: simulate(P, k) for k in sel}

with tabs[0]:
    fig, ax = plt.subplots(5, 1, figsize=(11, 15), sharex=True)
    any_r = next(iter(res.values()))[0]
    ax[0].plot(any_r["t"], any_r["ref"], "r--", lw=1.5, label="Reference")
    for k, (r, _) in res.items():
        ax[0].plot(r["t"], r["speed"], color=COL[k], lw=2, label=k)
        ax[1].plot(r["t"], r["ia"], color=COL[k], lw=2, label=k)
        ax[2].plot(r["t"], r["v"], color=COL[k], lw=2, label=k)
        ax[3].plot(r["t"], r["pos"], color=COL[k], lw=2, label=k)
    ax[4].plot(any_r["t"], any_r["load"], "k", lw=2, label="Load torque Td")
    if "MPC" in res:
        ax[4].plot(res["MPC"][0]["t"], res["MPC"][0]["dhat"], "tab:orange", ls="--", label="MPC estimate d̂")
    if use_W:
        ax[0].axhline(Wmax, color="gray", ls=":", label="Speed limit")
    for s in (1, -1):
        ax[1].axhline(s * Imax, color="gray", ls=":", label="Current limit" if s == 1 else None)
        ax[2].axhline(s * Vmax, color="gray", ls=":", label="Voltage limit" if s == 1 else None)
    titles = ["Speed ω (rad/s)", "Armature current ia (A)", "Applied voltage va (V)",
              "Angular position θ (rad)", "Disturbance torque (N·m)"]
    for a, tl in zip(ax, titles):
        a.set_ylabel(tl)
        a.grid(True, ls=":", alpha=0.6)
        a.legend(loc="upper right", fontsize=8)
    ax[4].set_xlabel("Time (s)")
    fig.tight_layout()
    st.pyplot(fig)

with tabs[1]:
    st.subheader("Performance & constraint satisfaction")
    df = pd.DataFrame({k: m for k, (_, m) in res.items()}).astype(str)
    st.dataframe(df, width="stretch", height=640)
    st.caption("Rise time = 10→90 %, settling band = ±2 %. For MPC with preview, the last Np·Ts seconds before an "
               "event are excluded from steady-state/overshoot windows (the controller acts early on purpose).")

with tabs[2]:
    st.subheader("Effect of prediction steps, control steps and priorities (MPC only)")
    c1, c2 = st.columns([1, 2])
    what = c1.selectbox("Sweep", ["Prediction steps Np", "Control steps Nc", "Control priority R (effort)",
                                  "Tracking priority Q"])
    default = {"Prediction steps Np": "5, 10, 20, 40", "Control steps Nc": "1, 2, 5, 10",
               "Control priority R (effort)": "0.001, 0.05, 0.5, 5", "Tracking priority Q": "0.1, 1, 10, 100"}[what]
    vals = c2.text_input("Values (comma separated, ≤ 6)", default)
    key = {"Prediction steps Np": "Np", "Control steps Nc": "Nc", "Control priority R (effort)": "R",
           "Tracking priority Q": "Q"}[what]
    if st.button("Run sweep"):
        try:
            values = [float(v) for v in vals.split(",") if v.strip()][:6]
        except ValueError:
            st.error("Enter numbers separated by commas.")
            st.stop()
        fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        rows = {}
        for v in values:
            Pi = dict(P)
            Pi[key] = int(v) if key in ("Np", "Nc") else v
            Pi["Nc"] = min(Pi["Nc"], Pi["Np"])
            r, m = simulate(Pi, "MPC")
            lab = f"{key}={Pi[key]}"
            ax[0].plot(r["t"], r["speed"], lw=2, label=lab)
            ax[1].plot(r["t"], r["ia"], lw=2, label=lab)
            ax[2].plot(r["t"], r["v"], lw=2, label=lab)
            rows[lab] = {k: m[k] for k in ["Rise time, 1st step (s)", "Settling time 2%, 1st step (s)",
                                           "Overshoot, 1st step (%)", "Disturbance: peak speed dip (rad/s)",
                                           "Disturbance: recovery time 2% (s)", "IAE (rad)",
                                           "Peak |current| (A)", "Mean compute time / step (ms)"]}
        any_r = r
        ax[0].plot(any_r["t"], any_r["ref"], "r--", label="Reference")
        for s in (1, -1):
            ax[1].axhline(s * Imax, color="gray", ls=":")
            ax[2].axhline(s * Vmax, color="gray", ls=":")
        for a, tl in zip(ax, ["Speed (rad/s)", "Current (A)", "Voltage (V)"]):
            a.set_ylabel(tl); a.grid(True, ls=":", alpha=0.6); a.legend(fontsize=8, loc="upper right")
        ax[2].set_xlabel("Time (s)")
        fig.tight_layout()
        st.pyplot(fig)
        st.dataframe(pd.DataFrame(rows).T.astype(str), width="stretch")

with tabs[3]:
    st.subheader("Model")
    st.latex(r"\dot i_a=\tfrac{1}{L_a}(v_a-R_a i_a-K_b\omega),\quad \dot\omega=\tfrac{1}{J}(K_t i_a-B\omega-T_d),\quad \dot\theta=\omega")
    st.latex(r"x_{k+1}=A_d x_k+B_d v_k+E_d d,\qquad x=[i_a,\ \omega,\ \theta]^T")
    st.subheader("MPC optimisation problem (solved every sample)")
    st.latex(r"\min_{\Delta v_{0..N_c-1},\,s}\ \sum_{k=1}^{N_p} Q\,(\omega_k-r_k)^2+\sum_{k=0}^{N_c-1}R\,\Delta v_k^2+\rho\sum_{k=1}^{N_p}(s_{i,k}^2+s_{\omega,k}^2)")
    st.latex(r"\text{s.t. } x_{k+1}=A_dx_k+B_dv_k+E_d\hat d,\quad v_k=v_{k-1}+\Delta v_k\ (\Delta v_k=0,\ k\ge N_c)")
    st.latex(r"|v_k|\le V_{max}\ (\text{hard}),\quad |i_{a,k}|\le I_{max}+s_{i,k},\quad |\omega_k|\le\omega_{max}+s_{\omega,k},\quad s\ge 0")
    st.markdown(
        "- **Np** = prediction steps, **Nc** = control adjustment (free move) steps; moves after Nc are frozen.\n"
        "- **Priorities**: Q (tracking) vs R (penalty on voltage changes) vs ρ (how strictly state limits are enforced).\n"
        "- **Offset-free tracking**: the load torque is estimated by \\(\\hat d_{k}=\\hat d_{k-1}+L_d\\,(\\omega_k-\\hat\\omega_{k|k-1})/E_{d,\\omega}\\) "
        "and held constant over the horizon, giving integral-like action.\n"
        "- **Voltage** is a hard input constraint; **current/speed** are soft (slack) so the QP is always feasible.\n"
        "- **PI/PID** use the same voltage saturation with clamping anti-windup and a filtered derivative, but have "
        "no knowledge of current/speed limits."
    )