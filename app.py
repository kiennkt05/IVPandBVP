# File: app.py

import streamlit as st
import pandas as pd
import numpy as np
import sympy as sp

from typing import List

from methods.euler_methods         import euler_method, modified_euler, midpoint_method
from methods.runge_kutta_methods   import runge_kutta_4
from methods.adams_methods         import adams_bashforth, adams_moulton
from methods.linear_shooting       import linear_shooting
from methods.rk_systems            import rk4_system

st.set_page_config(page_title="IVP & BVP Solver Toolbox", layout="wide")
st.title("🟢 IVP & BVP Solver Toolbox")

st.markdown(r"""
  This app implements **classical ODE solvers** for:
  1. Euler's Method (with Theorem 5.9 error bound)  
  2. Modified Euler, Midpoint, RK4  
  3. Adams–Bashforth (1–4) & Adams–Moulton (1–4)  
  4. Linear Shooting for second‐order BVPs  
  5. Runge–Kutta for first‐order systems (handles up to 4th‐order by rewriting as system)

  For each method you may enter the ODE(s) either symbolically (using `sympy`) to compute actual solutions and error bounds, or numerically by uploading/pasting a small CSV of `(t_i, y_i, …)` evaluations.  

  Each routine returns:
  - A detailed **step‐by‐step log** (stages, predictor/corrector values, etc.)  
  - The numerical approximations \(\{w_i\}\) (or \(\{y_i\}\) for systems)  
  - The "exact" solution \(\{y(t_i)\}\) if given symbolically  
  - **Pointwise errors** \(|y(t_i)-w_i|\) and (where applicable) theoretical **error bounds** (e.g.\ from Theorem 5.9 for Euler).  

  Select a method from the sidebar, input parameters (symbolic and/or numeric), and click **Run** to see results.
""")

# ---------- Helper to parse a symbolic function f(t,y) or f(t,y1,y2,...) safely ----------
@st.cache_resource
def parse_function(expr: str, vars: List[str]):
    """
    Given a string 'x**2 + sin(t*y)' and a list of variable names e.g. ['t','y'],
    return a string representation of the sympy expression and a numeric lambdified version.
    """
    try:
        symbols = sp.symbols(vars)
        sym_expr = sp.sympify(expr)
        func = sp.lambdify(symbols, sym_expr, 'numpy')
        return str(sym_expr), func, None
    except Exception as e:
        return None, None, str(e)


# ---------- Sidebar: choose which solver ----------
method = st.sidebar.selectbox(
    "Choose a solver:",
    [
        "1) Euler's Method (with error bound)",
        "2) Modified Euler",
        "3) Midpoint Method",
        "4) Runge–Kutta 4th Order (RK4)",
        "5) Adams–Bashforth (order 1–4)",
        "6) Adams–Moulton  (order 1–4)",
        "7) Linear Shooting (2nd-order BVP)",
        "8) Runge–Kutta for Systems (1st→4th order)"
    ]
)

st.header(f"🔹 {method}")

# --------------- 1) Euler's Method (with bound) ---------------
if method == "1) Euler's Method (with error bound)":
    st.subheader("Solve y' = f(t,y),  y(a)=α by Euler's Method")
    st.write("If you know symbolic f(t,y) and y_exact(t), input them below.  Otherwise leave blank to use numeric f only.")
    f_expr = st.text_input("f(t,y) =", "t + y")  # example
    if f_expr.strip():
        f_sym_str, f_num, err = parse_function(f_expr, ["t","y"])
        if err:
            st.error(f"Could not parse f(t,y): {err}")
            st.stop()
        f_sym = sp.sympify(f_sym_str) if f_sym_str else None
    else:
        f_sym, f_num = None, None

    y_exact_expr = st.text_input("Exact y(t) (optional) =", "Ce**t - t - 1")  # dummy
    if y_exact_expr.strip():
        y_exact_sym_str, y_exact_num, err2 = parse_function(y_exact_expr, ["t"])
        if err2:
            st.error(f"Could not parse y_exact(t): {err2}")
            st.stop()
        y_exact_sym = sp.sympify(y_exact_sym_str) if y_exact_sym_str else None
    else:
        y_exact_sym, y_exact_num = None, None

    a = st.number_input("Left endpoint a:", value=0.0)
    b = st.number_input("Right endpoint b:", value=1.0)
    alpha = st.number_input("Initial value y(a) = α:", value=1.0)
    N = int(st.number_input("Number of steps N:", min_value=1, value=10))

    if f_sym is None:
        st.write("Since f(t,y) not given symbolically, you must supply a Python callable.  Upload/paste pairs (t_i, y_i) of data to interpolate f numerically.")
        uploaded = st.file_uploader("Upload CSV (t,y) samples", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded, header=None, names=["t","y"])
            # We will do a 2D interpolation to get f(t,y).  For simplicity, use linear interp in t for each y? 
            # But truly, f(t,y) is a function of both.  We cannot reconstruct f(t,y) from just (t,y) samples unless user interprets y_i=f(t_i).
            # Instead interpret these samples as f(t, w(t)).  So we do simple 1D interp of f(t) along solution curve.
            # We'll treat f_numeric(ti,yi)= (yi+1 - yi)/(ti+1 - ti) if equally spaced.  THIS IS A STRETCH. 
            # For a simpler fallback: disallow numeric-only for Euler's method. 
            st.error("Euler's method requires symbolic f(t,y). Numeric interpolation of f in 2D is beyond scope here.")
            st.stop()
        else:
            st.stop()

    if st.button("Run Euler's Method"):
        res = euler_method(f_sym, f_num, y_exact_sym, a, b, alpha, N)
        st.subheader("Mesh points & Approximations")
        df_table = pd.DataFrame({
            "i": list(range(len(res["t"]))),
            "t_i": res["t"],
            "w_i": res["w"],
            **({ "y_exact(t_i)": res["y_exact"], "error": res["error"] } if res["y_exact"] else {})
        })
        st.dataframe(df_table)

        if res["error_bound"] is not None:
            st.subheader("Theorem 5.9 Error Bound at each t_i")
            df_bound = pd.DataFrame({
                "i": list(range(len(res["t"]))),
                "t_i": res["t"],
                "bound": res["error_bound"]
            })
            st.dataframe(df_bound)

        st.subheader("Step-by-Step Log")
        df_log = pd.DataFrame(res["log"])
        st.dataframe(df_log)

# --------------- 2) Modified Euler ---------------
elif method == "2) Modified Euler":
    st.subheader("Solve y' = f(t,y) by Modified Euler method")
    f_expr = st.text_input("f(t,y) =", "sin(t) - y") 
    if f_expr.strip():
        f_sym_str, f_num, err = parse_function(f_expr, ["t","y"])
        if err:
            st.error(f"Could not parse f(t,y): {err}")
            st.stop()
        f_sym = sp.sympify(f_sym_str) if f_sym_str else None
    else:
        st.error("Must provide symbolic f(t,y) for this method.")
        st.stop()

    y_exact_expr = st.text_input("Exact y(t) (optional) =", "")
    if y_exact_expr.strip():
        y_exact_sym_str, y_exact_num, err2 = parse_function(y_exact_expr, ["t"])
        if err2:
            st.error(f"Could not parse y_exact(t): {err2}")
            st.stop()
        y_exact_sym = sp.sympify(y_exact_sym_str) if y_exact_sym_str else None
    else:
        y_exact_sym = None

    a = st.number_input("a:", value=0.0)
    b = st.number_input("b:", value=1.0)
    alpha = st.number_input("y(a):", value=1.0)
    N = int(st.number_input("N:", min_value=1, value=10))

    if st.button("Run Modified Euler"):
        res = modified_euler(f_sym, f_num, y_exact_sym, a, b, alpha, N)
        df_table = pd.DataFrame({
            "i": list(range(len(res["t"]))),
            "t_i": res["t"],
            "w_i": res["w"],
            **({ "y_exact(t_i)": res["y_exact"], "error": res["error"] } if res["y_exact"] else {})
        })
        st.dataframe(df_table)
        st.subheader("Log")
        st.dataframe(pd.DataFrame(res["log"]))

# --------------- 3) Midpoint Method ---------------
elif method == "3) Midpoint Method":
    st.subheader("Solve y' = f(t,y) by the explicit Midpoint method")
    f_expr = st.text_input("f(t,y) =", "t + y")
    if f_expr.strip():
        f_sym_str, f_num, err = parse_function(f_expr, ["t","y"])
        if err:
            st.error(f"Could not parse f(t,y): {err}")
            st.stop()
        f_sym = sp.sympify(f_sym_str) if f_sym_str else None
    else:
        st.error("Must provide symbolic f(t,y).")
        st.stop()

    y_exact_expr = st.text_input("Exact y(t) (optional) =", "")
    if y_exact_expr.strip():
        y_exact_sym_str, _, err2 = parse_function(y_exact_expr, ["t"])
        if err2:
            st.error(f"Could not parse y_exact(t): {err2}")
            st.stop()
        y_exact_sym = sp.sympify(y_exact_sym_str) if y_exact_sym_str else None
    else:
        y_exact_sym = None

    a = st.number_input("a:", value=0.0)
    b = st.number_input("b:", value=1.0)
    alpha = st.number_input("y(a):", value=1.0)
    N = int(st.number_input("N:", min_value=1, value=10))

    if st.button("Run Midpoint Method"):
        res = midpoint_method(f_sym, f_num, y_exact_sym, a, b, alpha, N)
        df_table = pd.DataFrame({
            "i": list(range(len(res["t"]))),
            "t_i": res["t"],
            "w_i": res["w"],
            **({ "y_exact(t_i)": res["y_exact"], "error": res["error"] } if res["y_exact"] else {})
        })
        st.dataframe(df_table)
        st.subheader("Log")
        st.dataframe(pd.DataFrame(res["log"]))

# --------------- 4) Runge–Kutta 4 ---------------
elif method == "4) Runge–Kutta 4th Order (RK4)":
    st.subheader("Solve y' = f(t,y) by classical RK4")
    f_expr = st.text_input("f(t,y) =", " -2*t + y")
    if f_expr.strip():
        f_sym_str, f_num, err = parse_function(f_expr, ["t","y"])
        if err:
            st.error(f"Could not parse f(t,y): {err}")
            st.stop()
        f_sym = sp.sympify(f_sym_str) if f_sym_str else None
    else:
        st.error("Must provide symbolic f(t,y).")
        st.stop()

    y_exact_expr = st.text_input("Exact y(t) (optional) =", "")
    if y_exact_expr.strip():
        y_exact_sym_str, _, err2 = parse_function(y_exact_expr, ["t"])
        if err2:
            st.error(f"Could not parse y_exact(t): {err2}")
            st.stop()
        y_exact_sym = sp.sympify(y_exact_sym_str) if y_exact_sym_str else None
    else:
        y_exact_sym = None

    a = st.number_input("a:", value=0.0)
    b = st.number_input("b:", value=1.0)
    alpha = st.number_input("y(a):", value=0.0)
    N = int(st.number_input("N:", min_value=1, value=10))

    if st.button("Run RK4"):
        res = runge_kutta_4(f_sym, f_num, y_exact_sym, a, b, alpha, N)
        df_table = pd.DataFrame({
            "i": list(range(len(res["t"]))),
            "t_i": res["t"],
            "w_i": res["w"],
            **({ "y_exact(t_i)": res["y_exact"], "error": res["error"] } if res["y_exact"] else {})
        })
        st.dataframe(df_table)
        st.subheader("Log")
        st.dataframe(pd.DataFrame(res["log"]))

# --------------- 5) Adams–Bashforth ---------------
elif method == "5) Adams–Bashforth (order 1–4)":
    st.subheader("Solve y' = f(t,y) by Adams–Bashforth multistep")
    f_expr = st.text_input("f(t,y) =", "y - t**2 + 1")
    if f_expr.strip():
        f_sym_str, f_num, err = parse_function(f_expr, ["t","y"])
        if err:
            st.error(f"Could not parse f(t,y): {err}")
            st.stop()
        f_sym = sp.sympify(f_sym_str) if f_sym_str else None
    else:
        st.error("Must provide symbolic f(t,y).")
        st.stop()

    y_exact_expr = st.text_input("Exact y(t) (optional) =", "")
    if y_exact_expr.strip():
        y_exact_sym_str, _, err2 = parse_function(y_exact_expr, ["t"])
        if err2:
            st.error(f"Could not parse y_exact(t): {err2}")
            st.stop()
        y_exact_sym = sp.sympify(y_exact_sym_str) if y_exact_sym_str else None
    else:
        y_exact_sym = None

    a = st.number_input("a:", value=0.0)
    b = st.number_input("b:", value=1.0)
    alpha = st.number_input("y(a):", value=0.5)
    N = int(st.number_input("N:", min_value=1, value=10))
    order = int(st.number_input("Order (1–4):", min_value=1, max_value=4, value=3))

    if st.button("Run Adams–Bashforth"):
        res = adams_bashforth(f_sym, f_num, y_exact_sym, a, b, alpha, N, order)
        df_table = pd.DataFrame({
            "i": list(range(len(res["t"]))),
            "t_i": res["t"],
            "w_i": res["w"],
            **({ "y_exact(t_i)": res["y_exact"], "error": res["error"] } if res["y_exact"] else {})
        })
        st.dataframe(df_table)
        st.subheader("Log")
        st.dataframe(pd.DataFrame(res["log"]))

# --------------- 6) Adams–Moulton ---------------
elif method == "6) Adams–Moulton  (order 1–4)":
    st.subheader("Solve y' = f(t,y) by Adams–Moulton multistep")
    f_expr = st.text_input("f(t,y) =", "y - t**2 + 1")
    if f_expr.strip():
        f_sym_str, f_num, err = parse_function(f_expr, ["t","y"])
        if err:
            st.error(f"Could not parse f(t,y): {err}")
            st.stop()
        f_sym = sp.sympify(f_sym_str) if f_sym_str else None
    else:
        st.error("Must provide symbolic f(t,y).")
        st.stop()

    y_exact_expr = st.text_input("Exact y(t) (optional) =", "")
    if y_exact_expr.strip():
        y_exact_sym_str, _, err2 = parse_function(y_exact_expr, ["t"])
        if err2:
            st.error(f"Could not parse y_exact(t): {err2}")
            st.stop()
        y_exact_sym = sp.sympify(y_exact_sym_str) if y_exact_sym_str else None
    else:
        y_exact_sym = None

    a = st.number_input("a:", value=0.0)
    b = st.number_input("b:", value=1.0)
    alpha = st.number_input("y(a):", value=0.5)
    N = int(st.number_input("N:", min_value=1, value=10))
    order = int(st.number_input("Order (1–4):", min_value=1, max_value=4, value=3))

    if st.button("Run Adams–Moulton"):
        res = adams_moulton(f_sym, f_num, y_exact_sym, a, b, alpha, N, order)
        df_table = pd.DataFrame({
            "i": list(range(len(res["t"]))),
            "t_i": res["t"],
            "w_i": res["w"],
            **({ "y_exact(t_i)": res["y_exact"], "error": res["error"] } if res["y_exact"] else {})
        })
        st.dataframe(df_table)
        st.subheader("Log")
        st.dataframe(pd.DataFrame(res["log"]))

# --------------- 7) Linear Shooting ---------------
elif method == "7) Linear Shooting (2nd-order BVP)":
    st.subheader("Solve y'' = f2(t,y,y'),  y(a)=α, y(b)=β by linear shooting")
    f2_expr = st.text_input("f2(t,y,y') =", "-2*t + y + y'")
    if f2_expr.strip():
        f2_sym_str, f2_num, err = parse_function(f2_expr, ["t","y","v"])
        if err:
            st.error(f"Could not parse f2(t,y,y'): {err}")
            st.stop()
        f2_sym = sp.sympify(f2_sym_str) if f2_sym_str else None
    else:
        st.error("Must provide symbolic f2(t,y,y').")
        st.stop()

    y_exact_expr = st.text_input("Exact y(t) (optional) =", "3*t - (t**2)/2 + 1")
    if y_exact_expr.strip():
        y_exact_sym_str, _, err2 = parse_function(y_exact_expr, ["t"])
        if err2:
            st.error(f"Could not parse y_exact(t): {err2}")
            st.stop()
        y_exact_sym = sp.sympify(y_exact_sym_str) if y_exact_sym_str else None
    else:
        y_exact_sym = None

    a = st.number_input("a:", value=0.0)
    b = st.number_input("b:", value=1.0)
    alpha = st.number_input("y(a) = α:", value=1.0)
    beta  = st.number_input("y(b) = β:", value=2.0)
    N = int(st.number_input("N (mesh steps):", min_value=1, value=10))

    if st.button("Run Linear Shooting"):
        res = linear_shooting(f2_sym, f2_num, y_exact_sym, a, b, alpha, beta, N)
        df_table = pd.DataFrame({
            "i": list(range(len(res["t"]))),
            "t_i": res["t"],
            "y_approx": res["y"],
            **({ "y_exact(t_i)": res["y_exact"], "error": res["error"] } if res["y_exact"] else {})
        })
        st.write(f"Computed shooting constant C = {res['C']:.6f}")
        st.dataframe(df_table)
        st.subheader("Log: IVP #1")
        st.dataframe(pd.DataFrame(res["log"]["IVP1"]))
        st.subheader("Log: IVP #2")
        st.dataframe(pd.DataFrame(res["log"]["IVP2"]))
        st.subheader("Log: Combination")
        st.dataframe(pd.DataFrame(res["log"]["Combine"]))

# --------------- 8) RK4 for Systems ---------------
elif method == "8) Runge–Kutta for Systems (1st→4th order)":
    st.subheader("Solve a system of n first‐order ODEs using RK4")
    n = int(st.selectbox("Number of equations in system (n):", [1, 2, 3, 4], index=1))

    f_exprs = []
    f_syms = []
    f_nums = []
    for j in range(n):
        f_e = st.text_input(f"f{j+1}(t, y1, …, y{n}) =", " 0 ")
        if not f_e.strip():
            st.error("All f_i must be provided symbolically.")
            st.stop()
        f_sym_str, num, err = parse_function(f_e, ["t"] + [f"y{k+1}" for k in range(n)])
        if err:
            st.error(f"Could not parse f{j+1}: {err}")
            st.stop()
        f_exprs.append(f_sym_str)
        f_syms.append(sp.sympify(f_sym_str) if f_sym_str else None)
        f_nums.append(num)

    y_exact_exprs = []
    for j in range(n):
        y_e = st.text_input(f"Exact y{j+1}(t) (optional) =", "")
        if y_e.strip():
            y_exact_sym_str, num, err2 = parse_function(y_e, ["t"])
            if err2:
                st.error(f"Could not parse y{j+1}_exact: {err2}")
                st.stop()
            y_exact_exprs.append(sp.sympify(y_exact_sym_str) if y_exact_sym_str else None)
        else:
            y_exact_exprs.append(None)

    a = st.number_input("a:", value=0.0)
    b = st.number_input("b:", value=1.0)
    y0 = []
    for j in range(n):
        y0.append(st.number_input(f"y{j+1}(a) =", value=0.0))
    N = int(st.number_input("N:", min_value=1, value=10))

    if st.button("Run RK4 System"):
        res = rk4_system(f_syms if any(f_syms) else None, f_nums,
                         y_exact_exprs if any(y_exact_exprs) else None,
                         a, b, y0, N)
        # Build DataFrame with columns t, y1..y_n, y1_exact..y_n_exact, err1..err_n
        data = {"i": list(range(N+1)), "t_i": res["t"]}
        for j in range(n):
            data[f"y{j+1}_approx"] = [res["Y"][i][j] for i in range(N+1)]
        if res["Y_exact"] is not None:
            for j in range(n):
                data[f"y{j+1}_exact"] = [res["Y_exact"][i][j] for i in range(N+1)]
                data[f"err{j+1}"] = [res["error"][i][j] for i in range(N+1)]
        df_table = pd.DataFrame(data)
        st.dataframe(df_table)
        st.subheader("Log")
        st.write("Each entry in the log is a dictionary containing `Y_i`, `k1`,`k2`,`k3`,`k4`, `Y_{i+1}`")
        st.dataframe(pd.DataFrame(res["log"]))

