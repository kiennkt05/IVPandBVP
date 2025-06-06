# File: methods/runge_kutta_methods.py

from typing import Callable, List, Dict, Any, Optional
import sympy as sp
import numpy as np

def runge_kutta_4(
    f_sym: Optional[sp.Expr],
    f_numeric: Callable[[float, float], float],
    y_exact_sym: Optional[sp.Expr],
    a: float,
    b: float,
    alpha: float,
    N: int
) -> Dict[str, Any]:
    """
    Classic fourth‐order Runge‐Kutta (RK4) for y' = f(t,y):
      k1 = f(t_i,       w_i)
      k2 = f(t_i + h/2, w_i + (h/2) k1)
      k3 = f(t_i + h/2, w_i + (h/2) k2)
      k4 = f(t_i + h,   w_i + h k3)
      w_{i+1} = w_i + (h/6)(k1 + 2k2 + 2k3 + k4)

    INPUT same as Euler. OUTPUT:
      {
        "t": [...],
        "w": [...],
        "y_exact": [...],  # if available
        "error": [...],    # if available
        "log": [
          {
            "i":i, "t_i":ti, "w_i":wi,
            "k1":k1, "k2":k2, "k3":k3, "k4":k4,
            "w_{i+1}": w_{i+1}
          }, ...
        ]
      }
    """

    h = (b - a)/N
    t = [a + i*h for i in range(N+1)]
    w = [0.0]*(N+1)
    w[0] = alpha
    log: List[Dict[str, Any]] = []

    for i in range(N):
        ti = t[i]
        wi = w[i]
        k1 = f_numeric(ti, wi)
        k2 = f_numeric(ti + h/2, wi + h*k1/2)
        k3 = f_numeric(ti + h/2, wi + h*k2/2)
        k4 = f_numeric(ti + h, wi + h*k3)
        wi1 = wi + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        w[i+1] = wi1
        log.append({
            "i": i,
            "t_i": ti,
            "w_i": wi,
            "k1": k1,
            "k2": k2,
            "k3": k3,
            "k4": k4,
            "w_{i+1}": wi1
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
