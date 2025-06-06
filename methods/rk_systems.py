# File: methods/rk_systems.py

from typing import Callable, List, Dict, Any, Optional
import numpy as np
import sympy as sp

def rk4_system(
    f_sym_list: Optional[List[sp.Expr]],
    f_numeric_list: List[Callable[..., float]],
    y_exact_sym_list: Optional[List[sp.Expr]],
    a: float,
    b: float,
    y0: List[float],
    N: int
) -> Dict[str, Any]:
    """
    Runge-Kutta 4th order for a first-order system of n equations:
      y'_1 = f1(t, y1, y2, …, yn)
      y'_2 = f2(t, y1, y2, …, yn)
      ...
      y'_n = fn(t, y1, y2, …, yn)
    with initial condition Y(a) = [y1(a), y2(a), …, yn(a)] = y0 (length n), on [a,b] with N steps.

    We store each Y_i = [y1_i, …, yn_i] at t_i.

    If `y_exact_sym_list` is provided (list of sympy expressions for y1(t), …,y_n(t)),
    we compute pointwise errors.

    INPUT:
      - f_sym_list        : list of sympy expressions [f1(t, y1..yn), …, fn(t,y1..yn)] or None
      - f_numeric_list    : list of Python callables f_i(t, y_vector) → float
      - y_exact_sym_list  : list of sympy expressions for exact solutions (or None)
      - a, b, N           : interval endpoints and number of steps
      - y0                : list of length n of initial values at t=a

    OUTPUT:
      {
        "t": [...],
        "Y": [ [y1_i,...,y_n_i] for i in 0..N ],
        "Y_exact": [ [y1_exact(t_i),...] ] or None,
        "error":  [ [|y_exact - y_i|...] ] or None,
        "log": [ { "i":i, "t_i":t_i, "Y_i":..., "k1":[...], "k2":[...], "k3":[...], "k4":[...], "Y_{i+1}":[...] }, ... ]
      }

    NOTE: Up to n=4 is typical (we can handle general n though).
    """

    h = (b - a)/N
    t = [a + i*h for i in range(N+1)]
    n = len(y0)
    # Y is a list of lists, each of length n
    Y = [ [0.0]*n for _ in range(N+1) ]
    Y[0] = y0.copy()

    log: List[Dict[str, Any]] = []

    def vector_f(ti: float, Yi: List[float]) -> List[float]:
        return [ f_numeric_list[j](ti, *Yi) for j in range(n) ]

    for i in range(N):
        ti = t[i]
        Yi = Y[i]
        # k1
        f1 = vector_f(ti, Yi)
        k1 = [ h * f1[j] for j in range(n) ]
        # k2
        Yi_k1_half = [ Yi[j] + 0.5*k1[j] for j in range(n) ]
        f2 = vector_f(ti + h/2, Yi_k1_half)
        k2 = [ h * f2[j] for j in range(n) ]
        # k3
        Yi_k2_half = [ Yi[j] + 0.5*k2[j] for j in range(n) ]
        f3 = vector_f(ti + h/2, Yi_k2_half)
        k3 = [ h * f3[j] for j in range(n) ]
        # k4
        Yi_k3 = [ Yi[j] + k3[j] for j in range(n) ]
        f4 = vector_f(ti + h, Yi_k3)
        k4 = [ h * f4[j] for j in range(n) ]
        # Combine
        Yi1 = [ Yi[j] + (1/6)*(k1[j] + 2*k2[j] + 2*k3[j] + k4[j]) for j in range(n) ]
        Y[i+1] = Yi1

        log.append({
            "i": i,
            "t_i": ti,
            "Y_i": Yi.copy(),
            "k1": k1.copy(),
            "k2": k2.copy(),
            "k3": k3.copy(),
            "k4": k4.copy(),
            "Y_{i+1}": Yi1.copy()
        })

    # Compute exact and error if available
    if y_exact_sym_list is not None:
        t_sym = sp.symbols('t')
        y_exact_funcs = [ sp.lambdify(t_sym, expr, 'numpy') for expr in y_exact_sym_list ]
        Y_exact = []
        for ti in t:
            Y_exact.append([ float(f(ti)) for f in y_exact_funcs ])
        error = [
            [ abs(Y_exact[i][j] - Y[i][j]) for j in range(n) ]
            for i in range(N+1)
        ]
    else:
        Y_exact = None
        error = None

    return {
        "t": t,
        "Y": Y,
        "Y_exact": Y_exact,
        "error": error,
        "log": log
    }
