# File: methods/euler_methods.py

from typing import Callable, List, Dict, Any, Optional
import numpy as np
import sympy as sp

def euler_method(
    f_sym: Optional[sp.Expr],
    f_numeric: Callable[[float, float], float],
    y_exact_sym: Optional[sp.Expr],
    a: float,
    b: float,
    alpha: float,
    N: int
) -> Dict[str, Any]:
    """
    Euler's Method for y' = f(t, y),  t in [a,b], y(a)=alpha, with N steps.
    Also computes actual errors (if exact y(t) given) and the error bound from Theorem 5.9.

    Theorem 5.9 bound:
       | y(t_i) - w_i | ≤ (h M / (2 L)) [ e^{L (t_i - a)} - 1 ]
    where L = Lipschitz constant in y, M = max |y''(t)| on [a,b], h = (b-a)/N.

    INPUT:
      - f_sym       : sympy expression for f(t,y)  (or None if only numeric)
      - f_numeric   : a Python callable f(t,y) that returns float
      - y_exact_sym : sympy expression for exact y(t)  (or None)
      - a, b        : interval endpoints
      - alpha       : initial value y(a)
      - N           : number of steps

    OUTPUT dict:
      {
        "t"          : [t0,...,tN],
        "w"          : [w0,...,wN]  (Euler iterates),
        "y_exact"    : [y_exact(t0)...y_exact(tN)] or None,
        "error"      : [|y_exact(ti)-wi|...] or None,
        "error_bound": [ bound_i ... ] or None,
        "log"        : [ {"i": i, "t_i": ti, "w_i": wi, "f": f(ti,wi)} , ... ]
      }
    """

    # 1) Setup mesh
    h = (b - a) / N
    t = [a + i*h for i in range(N+1)]
    w = [0.0]*(N+1)
    w[0] = alpha

    # 2) If possible, form symbolic partial wrt y to get Lipschitz L
    if f_sym is not None:
        t_sym, y_sym = sp.symbols('t y')
        # We assume f_sym is a sympy expression in t_sym, y_sym
        # Lipschitz in y: take ∂f/∂y and bound its absolute value on [a,b] x [range_of_y]
        df_dy = sp.diff(f_sym, y_sym)
        df_dy_func = sp.lambdify((t_sym, y_sym), df_dy, 'numpy')
        # We will estimate L by sampling (t in [a,b], w in approximate [min,max] range)
        # First do a quick Euler pass with small step to see the range of y
        # But to be safe, we'll sample y over [a, alpha + M*(b-a)] ??? → instead do coarse grid:
        t_samples = np.linspace(a, b, 50)
        y_min = alpha; y_max = alpha
        # We will later update y_min,y_max once we have w_i
    else:
        df_dy = None

    # 3) Evaluate M = max |y''(t)| on [a,b], if y_exact given
    if y_exact_sym is not None:
        t_sym = sp.symbols('t')
        # First substitute any constants with their numeric values
        y_exact_sym = y_exact_sym.subs('Ce', 1.0)  # Replace Ce with 1.0 for evaluation
        ypp = sp.diff(y_exact_sym, t_sym, 2)
        ypp_func = sp.lambdify(t_sym, ypp, 'numpy')
        t_samples = np.linspace(a, b, 200)
        M_val = float(np.max(np.abs(ypp_func(t_samples))))
    else:
        M_val = None

    # 4) Main Euler iteration + collect log
    log: List[Dict[str, Any]] = []
    for i in range(N):
        ti = t[i]
        wi = w[i]
        fi = f_numeric(ti, wi)
        w[i+1] = wi + h * fi
        log.append({
            "i": i,
            "t_i": ti,
            "w_i": wi,
            "f(t_i,w_i)": fi,
            "w_{i+1}": w[i+1]
        })

    # 5) Compute exact solution values + pointwise errors if available
    if y_exact_sym is not None:
        t_sym = sp.symbols('t')
        y_exact_func = sp.lambdify(t_sym, y_exact_sym, 'numpy')
        y_exact = [ float(y_exact_func(ti)) for ti in t ]
        error = [ abs(y_exact[i] - w[i]) for i in range(N+1) ]
    else:
        y_exact = None
        error = None

    # 6) Compute Lipschitz constant L by sampling ∂f/∂y over a rectangle [a,b]×[min(w),max(w)]
    if f_sym is not None:
        # estimate y_min,y_max from computed w's + a margin
        w_arr = np.array(w)
        y_min = float(np.min(w_arr)) - 0.1*abs(np.min(w_arr)+1)  # a small cushion
        y_max = float(np.max(w_arr)) + 0.1*abs(np.max(w_arr)+1)
        t_samples = np.linspace(a, b, 50)
        y_samples = np.linspace(y_min, y_max, 50)
        L_candidate = 0.0
        for ts in t_samples:
            for ys in y_samples:
                val = abs(df_dy_func(ts, ys))
                if val > L_candidate:
                    L_candidate = val
        L_val = float(L_candidate)
    else:
        L_val = None

    # 7) Compute error‐bound at each t_i:  (h M/(2 L)) [ e^{L(t_i - a)} - 1 ]
    if (y_exact_sym is not None) and (f_sym is not None):
        if L_val == 0:
            # If ∂f/∂y = 0 everywhere, Euler is exact for linear f, bound = M*h*(ti - a)^something? → use limit L→0: bound = (M * h * (t_i - a)) / 2
            error_bound = [ (M_val * h * (t[i] - a) / 2.0)  for i in range(N+1) ]
        else:
            error_bound = []
            for i in range(N+1):
                ti = t[i]
                bound_i = (h * M_val / (2*L_val)) * ( np.exp(L_val*(ti - a)) - 1.0 )
                error_bound.append(bound_i)
    else:
        error_bound = None

    return {
        "t": t,
        "w": w,
        "y_exact": y_exact,
        "error": error,
        "error_bound": error_bound,
        "log": log
    }


def modified_euler(
    f_sym: Optional[sp.Expr],
    f_numeric: Callable[[float, float], float],
    y_exact_sym: Optional[sp.Expr],
    a: float,
    b: float,
    alpha: float,
    N: int
) -> Dict[str, Any]:
    """
    Modified‐Euler (Heun's) method for y' = f(t,y):
      w_{i+1} = w_i + (h/2)[ f(t_i,w_i) + f(t_{i+1}, w_i + h f(t_i,w_i)) ]

    INPUT same as Euler. OUTPUT similar structure:
      {
        "t": [...],
        "w": [...],
        "y_exact": [...],  # if available
        "error": [...],    # if available
        "log": [ {"i":i, "t_i":ti, "w_i":wi, "k1":k1, "k2":k2, "w_{i+1}":...}, ... ]
      }
    """

    h = (b - a) / N
    t = [a + i*h for i in range(N+1)]
    w = [0.0]*(N+1)
    w[0] = alpha
    log: List[Dict[str, Any]] = []

    for i in range(N):
        ti = t[i]
        wi = w[i]
        k1 = f_numeric(ti, wi)
        y_pred = wi + h * k1
        k2 = f_numeric(ti + h, y_pred)
        w[i+1] = wi + (h/2)*(k1 + k2)
        log.append({
            "i": i,
            "t_i": ti,
            "w_i": wi,
            "k1": k1,
            "k2": k2,
            "w_{i+1}": w[i+1]
        })

    if y_exact_sym is not None:
        t_sym = sp.symbols('t')
        y_exact_func = sp.lambdify(t_sym, y_exact_sym, 'numpy')
        y_exact = [float(y_exact_func(ti)) for ti in t]
        error = [abs(y_exact[i] - w[i]) for i in range(N+1)]
    else:
        y_exact = None
        error = None

    return {
        "t": t,
        "w": w,
        "y_exact": y_exact,
        "error": error,
        "log": log
    }


def midpoint_method(
    f_sym: Optional[sp.Expr],
    f_numeric: Callable[[float, float], float],
    y_exact_sym: Optional[sp.Expr],
    a: float,
    b: float,
    alpha: float,
    N: int
) -> Dict[str, Any]:
    """
    Midpoint method (explicit) for y' = f(t,y):
      w_{i+1} = w_i + h f(t_i + h/2, w_i + (h/2) f(t_i, w_i))

    OUTPUT dictionary as above, with "log" entries:
      {"i":i, "t_i":ti, "w_i":wi, "k1":k1, "t_mid":tMid, "y_mid":yMid, "k2":k2, "w_{i+1}":...}
    """

    h = (b - a) / N
    t = [a + i*h for i in range(N+1)]
    w = [0.0]*(N+1)
    w[0] = alpha
    log: List[Dict[str, Any]] = []

    for i in range(N):
        ti = t[i]
        wi = w[i]
        k1 = f_numeric(ti, wi)
        t_mid = ti + h/2
        y_mid = wi + (h/2)*k1
        k2 = f_numeric(t_mid, y_mid)
        w[i+1] = wi + h * k2
        log.append({
            "i": i,
            "t_i": ti,
            "w_i": wi,
            "k1": k1,
            "t_mid": t_mid,
            "y_mid": y_mid,
            "k2": k2,
            "w_{i+1}": w[i+1]
        })

    if y_exact_sym is not None:
        t_sym = sp.symbols('t')
        y_exact_func = sp.lambdify(t_sym, y_exact_sym, 'numpy')
        y_exact = [float(y_exact_func(ti)) for ti in t]
        error = [abs(y_exact[i] - w[i]) for i in range(N+1)]
    else:
        y_exact = None
        error = None

    return {
        "t": t,
        "w": w,
        "y_exact": y_exact,
        "error": error,
        "log": log
    }
