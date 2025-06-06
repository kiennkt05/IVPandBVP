# File: methods/linear_shooting.py

from typing import Callable, List, Dict, Any, Optional
import numpy as np
import sympy as sp

def linear_shooting(
    f2_sym: Optional[sp.Expr],
    f2_numeric: Callable[[float, float, float], float],
    y_exact_sym: Optional[sp.Expr],
    a: float,
    b: float,
    alpha: float,
    beta: float,
    N: int
) -> Dict[str, Any]:
    """
    Linear shooting method for y'' = f2(t, y, y'),  with boundary conditions y(a)=alpha, y(b)=beta.
    We convert the BVP to two IVPs:
      (1) y_1'' = f2(t, y_1, y_1'),   y_1(a)=alpha, y_1'(a)=0
      (2) y_2'' = f2(t, y_2, y_2'),   y_2(a)=0,     y_2'(a)=1

    Then the general solution is:  y(t) = y_1(t) + C * y_2(t)
    We pick C so that y(b) = beta ⇒ C = (beta - y_1(b)) / y_2(b).

    We approximate each of the two IVPs by, say, RK4 on the first‐order system:
       Let v = y', then
       y' = v,
       v' = f2(t, y, v).
    Use N steps on [a,b] for both.

    OUTPUT:
      {
        "t": [...],
        "y": [...],       # combined solution y_i = y1_i + C y2_i
        "y_exact": [...], # if available
        "error": [...],   # if available
        "C": C_value,
        "log": {
          "IVP1": [ {i,t_i,y1_i,v1_i} ],
          "IVP2": [ {i,t_i,y2_i,v2_i} ],
          "combine": [{"C": C_value, "y_i": y_i, "i":i}, ... ]
        }
      }
    """

    h = (b - a)/N
    t = [a + i*h for i in range(N+1)]
    # Containers for y1, v1, y2, v2
    y1 = [0.0]*(N+1); v1 = [0.0]*(N+1)
    y2 = [0.0]*(N+1); v2 = [0.0]*(N+1)
    # Initial conditions:
    y1[0] = alpha; v1[0] = 0.0
    y2[0] = 0.0;    v2[0] = 1.0

    log1: List[Dict[str, Any]] = []
    log2: List[Dict[str, Any]] = []

    # Helper: single RK4 step for system y'=v, v'=f2(t,y,v)
    def rk4_step(yi, vi, ti, h):
        k1y = vi
        k1v = f2_numeric(ti, yi, vi)
        k2y = vi + (h/2)*k1v
        k2v = f2_numeric(ti + h/2, yi + (h/2)*k1y, vi + (h/2)*k1v)
        k3y = vi + (h/2)*k2v
        k3v = f2_numeric(ti + h/2, yi + (h/2)*k2y, vi + (h/2)*k2v)
        k4y = vi + h*k3v
        k4v = f2_numeric(ti + h, yi + h*k3y, vi + h*k3v)
        yi1 = yi + (h/6)*(k1y + 2*k2y + 2*k3y + k4y)
        vi1 = vi + (h/6)*(k1v + 2*k2v + 2*k3v + k4v)
        return yi1, vi1, k1y, k1v, k2y, k2v, k3y, k3v, k4y, k4v

    # Integrate IVP #1
    for i in range(N):
        ti = t[i]
        yi, vi = y1[i], v1[i]
        (y1_i1, v1_i1,
         k1y1, k1v1,
         k2y1, k2v1,
         k3y1, k3v1,
         k4y1, k4v1) = rk4_step(yi, vi, ti, h)
        y1[i+1] = y1_i1
        v1[i+1] = v1_i1
        log1.append({
            "i": i,
            "t_i": ti,
            "y1_i": yi,
            "v1_i": vi,
            "k1y1": k1y1, "k1v1": k1v1,
            "k2y1": k2y1, "k2v1": k2v1,
            "k3y1": k3y1, "k3v1": k3v1,
            "k4y1": k4y1, "k4v1": k4v1,
            "y1_{i+1}": y1_i1, "v1_{i+1}": v1_i1
        })

    # Integrate IVP #2
    for i in range(N):
        ti = t[i]
        yi, vi = y2[i], v2[i]
        (y2_i1, v2_i1,
         k1y2, k1v2,
         k2y2, k2v2,
         k3y2, k3v2,
         k4y2, k4v2) = rk4_step(yi, vi, ti, h)
        y2[i+1] = y2_i1
        v2[i+1] = v2_i1
        log2.append({
            "i": i,
            "t_i": ti,
            "y2_i": yi,
            "v2_i": vi,
            "k1y2": k1y2, "k1v2": k1v2,
            "k2y2": k2y2, "k2v2": k2v2,
            "k3y2": k3y2, "k3v2": k3v2,
            "k4y2": k4y2, "k4v2": k4v2,
            "y2_{i+1}": y2_i1, "v2_{i+1}": v2_i1
        })

    # Compute the constant C so that y(a)+C*y2(b) = beta => C = (beta - y1(b)) / y2(b)
    if abs(y2[-1]) < 1e-14:
        C = float('nan')  # degenerate
    else:
        C = (beta - y1[-1]) / y2[-1]

    # Combine to get the BVP solution
    y = [ y1[i] + C*y2[i] for i in range(N+1) ]

    # Actual solution if given
    if y_exact_sym is not None:
        t_sym = sp.symbols('t')
        y_exact_func = sp.lambdify(t_sym, y_exact_sym, 'numpy')
        y_exact = [ float(y_exact_func(ti)) for ti in t ]
        error = [ abs(y_exact[i] - y[i]) for i in range(N+1) ]
    else:
        y_exact = None
        error = None

    # Combine log: we will interleave them or just return as a dict
    log_combined = {
        "IVP1": log1,
        "IVP2": log2,
        "Combine": [
            {"i": i, "C": C, "y_i": y[i]} for i in range(N+1)
        ]
    }

    return {
        "t": t,
        "y": y,
        "C": C,
        "y_exact": y_exact,
        "error": error,
        "log": log_combined
    }
