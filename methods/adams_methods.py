# File: methods/adams_methods.py

from typing import Callable, List, Dict, Any, Optional
import sympy as sp
import numpy as np

def _euler_to_generate(
    f: Callable[[float, float], float],
    a: float,
    alpha: float,
    h: float,
    num_steps: int
) -> List[float]:
    """
    Utility: use simple Euler to generate starting values w0, w1, ... up to w_num_steps.
    """
    w0 = alpha
    w_vals = [w0]
    t = a
    for i in range(num_steps):
        w_next = w_vals[-1] + h * f(a + i*h, w_vals[-1])
        w_vals.append(w_next)
    return w_vals

def adams_bashforth(
    f_sym: Optional[sp.Expr],
    f_numeric: Callable[[float, float], float],
    y_exact_sym: Optional[sp.Expr],
    a: float,
    b: float,
    alpha: float,
    N: int,
    order: int
) -> Dict[str, Any]:
    """
    Adams–Bashforth explicit multistep method (1 ≤ order ≤ 4) for y' = f(t,y).
    We use Euler or RK4 to generate the first (order-1) values exactly.

    Coefficients (for uniform step h):
      AB1 (Euler):       w_{i+1} = w_i + h f_i
      AB2: w_{i+2} = w_{i+1} + (h/2) [ 3 f_{i+1} - f_i ]
      AB3: w_{i+3} = w_{i+2} + (h/12) [ 23 f_{i+2} - 16 f_{i+1} + 5 f_i ]
      AB4: w_{i+4} = w_{i+3} + (h/24) [ 55 f_{i+3} - 59 f_{i+2} + 37 f_{i+1} - 9 f_i ]

    INPUT:
      - f_sym, f_numeric, y_exact_sym, a,b,alpha,N  same as before
      - order: 1,2,3 or 4

    OUTPUT:
      {
        "t": [...],
        "w": [...],
        "y_exact": [...],  # if available
        "error": [...],    # if available
        "log": [ {"i":i, "t_i":t_i, "w_i":w_i, "f_i":f_i, ..., "w_{i+order}":...}, ... ]
      }
    """

    h = (b - a)/N
    t = [a + i*h for i in range(N+1)]
    w = [0.0]*(N+1)
    w[0] = alpha

    # We need first (order) points: we will use RK4 to generate w1..w_{order-1}
    # For simplicity, let's call runge_kutta_4 with N = order-1 over [a, a + (order-1)*h]
    # Actually, we want EXACT starting values if y_exact_sym is given; if so, set w[1..order] by exact y.
    if y_exact_sym is not None:
        t_sym = sp.symbols('t')
        y_exact_func = sp.lambdify(t_sym, y_exact_sym, 'numpy')
        for j in range(1, order):
            w[j] = float(y_exact_func(t[j]))
    else:
        # Fallback: use Euler (low accuracy) to generate
        w_gen = _euler_to_generate(f_numeric, a, alpha, h, order-1)
        for j in range(1, order):
            w[j] = w_gen[j]

    log: List[Dict[str, Any]] = []
    # Precompute f values at initial nodes:
    f_vals = [ f_numeric(ti, w_i) for ti, w_i in zip(t[:order], w[:order]) ]

    for i in range(order-1, N):
        # we now compute w_{i+1} (for order=1), or w_{i+2} if order=2, etc.
        if order == 1:
            # AB1 (Euler)  
            fi = f_numeric(t[i], w[i])
            w[i+1] = w[i] + h * fi
            log.append({
                "i": i,
                "t_i": t[i],
                "w_i": w[i],
                "f_i": fi,
                "w_{i+1}": w[i+1]
            })
            f_vals.append(f_numeric(t[i+1], w[i+1]))

        elif order == 2:
            # AB2: w_{i+1} known, we need i>=1 to compute w_{i+2}? Actually handle as i→ i+1:
            if i == 1:
                # We have w[0], w[1] from exact/Euler. We compute w[2].
                f0 = f_vals[0]
                f1 = f_vals[1]
                w[2] = w[1] + (h/2)*(3*f1 - f0)
                log.append({
                    "i": i-1,
                    "t_i": t[i-1],
                    "w_i": w[i-1],
                    "f_i": f0,
                    "f_{i+1}": f1,
                    "w_{i+2}": w[2]
                })
                f_vals.append(f_numeric(t[2], w[2]))
            else:
                # General step: we are at indices (i-1, i). We computed w[i] and f[i].
                f_i1 = f_vals[i]
                f_i0 = f_vals[i-1]
                w[i+1] = w[i] + (h/2)*(3*f_i1 - f_i0)
                log.append({
                    "i": i-1,
                    "t_i": t[i-1],
                    "w_i": w[i-1],
                    "f_{i-1}": f_i0,
                    "f_i": f_i1,
                    "w_{i+1}": w[i+1]
                })
                f_vals.append(f_numeric(t[i+1], w[i+1]))

        elif order == 3:
            # AB3: w_{i+1} = w_i + (h/12)(23 f_i - 16 f_{i-1} + 5 f_{i-2})
            if i < 2:
                continue  # skip until we have f_vals[0..2]
            f_im2 = f_vals[i-2]
            f_im1 = f_vals[i-1]
            f_i   = f_vals[i]
            w[i+1] = w[i] + (h/12)*(23*f_i - 16*f_im1 + 5*f_im2)
            log.append({
                "i": i-2,
                "t_i": t[i-2],
                "w_i": w[i-2],
                "f_{i-2}": f_im2,
                "f_{i-1}": f_im1,
                "f_i": f_i,
                "w_{i+1}": w[i+1]
            })
            f_vals.append(f_numeric(t[i+1], w[i+1]))

        elif order == 4:
            # AB4: w_{i+1} = w_i + (h/24)(55 f_i - 59 f_{i-1} + 37 f_{i-2} - 9 f_{i-3})
            if i < 3:
                continue
            f_im3 = f_vals[i-3]
            f_im2 = f_vals[i-2]
            f_im1 = f_vals[i-1]
            f_i   = f_vals[i]
            w[i+1] = w[i] + (h/24)*(55*f_i - 59*f_im1 + 37*f_im2 - 9*f_im3)
            log.append({
                "i": i-3,
                "t_i": t[i-3],
                "w_i": w[i-3],
                "f_{i-3}": f_im3,
                "f_{i-2}": f_im2,
                "f_{i-1}": f_im1,
                "f_i": f_i,
                "w_{i+1}": w[i+1]
            })
            f_vals.append(f_numeric(t[i+1], w[i+1]))
        else:
            raise ValueError("Adams–Bashforth order must be 1..4")

    # Now compute exact and error if available
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


def adams_moulton(
    f_sym: Optional[sp.Expr],
    f_numeric: Callable[[float, float], float],
    y_exact_sym: Optional[sp.Expr],
    a: float,
    b: float,
    alpha: float,
    N: int,
    order: int
) -> Dict[str, Any]:
    """
    Adams–Moulton implicit multistep methods (1 ≤ order ≤ 4) for y' = f(t,y).
    We use RK4 to generate starting values. Then solve the implicit equation (fixed‐point or single Newton iteration).
    
    AM1 (Trapezoidal):    w_{i+1} = w_i + (h/2)( f_i + f_{i+1} )
    AM2:                  w_{i+2} = w_{i+1} + (h/12)(5 f_{i+2} + 8 f_{i+1} - f_i)
    AM3:                  w_{i+3} = w_{i+2} + (h/24)(9 f_{i+3} + 19 f_{i+2} - 5 f_{i+1} + f_i)
    AM4:                  w_{i+4} = w_{i+3} + (h/720)(251 f_{i+4} + 646 f_{i+3} - 264 f_{i+2} + 106 f_{i+1} - 19 f_i)

    We solve the implicit equation for w_{i+order} using one Newton‐step:
      w_new = w_pred + correction.  (For simplicity we approximate by using f evaluated at predicted w.)
    
    INPUT & OUTPUT: same structure as adams_bashforth.
    """

    # First generate first (order) points using exact solution or RK4/Euler if no exact
    h = (b - a)/N
    t = [a + i*h for i in range(N+1)]
    w = [0.0]*(N+1)
    w[0] = alpha

    # Starting values:
    if y_exact_sym is not None:
        t_sym = sp.symbols('t')
        y_exact_func = sp.lambdify(t_sym, y_exact_sym, 'numpy')
        for j in range(1, order):
            w[j] = float(y_exact_func(t[j]))
    else:
        # Use RK4 to fill w[1..order-1]
        from methods.runge_kutta_methods import runge_kutta_4
        starter = runge_kutta_4(None, f_numeric, y_exact_sym, a, a+(order-1)*h, alpha, order-1)
        for j in range(1, order):
            w[j] = starter["w"][j]

    log: List[Dict[str, Any]] = []
    # f-values storage:
    f_vals = [f_numeric(ti, wi) for ti, wi in zip(t[:order], w[:order])]

    for i in range(order-1, N):
        if order == 1:
            # Trapezoidal: w_{i+1} = w_i + (h/2)( f_i + f_{i+1} )
            # Implicit since f_{i+1} depends on w_{i+1}. We'll do one predictor (Euler) then correct:
            # Predictor: w_pred = w_i + h f_i
            w_pred = w[i] + h * f_vals[i]
            f_pred = f_numeric(t[i+1], w_pred)
            w[i+1] = w[i] + (h/2)*(f_vals[i] + f_pred)
            log.append({
                "i": i,
                "t_i": t[i],
                "w_i": w[i],
                "f_i": f_vals[i],
                "w_pred": w_pred,
                "f_pred": f_pred,
                "w_{i+1}": w[i+1]
            })
            f_vals.append(f_numeric(t[i+1], w[i+1]))

        elif order == 2:
            # AM2: w_{i+2} = w_{i+1} + (h/12)(5 f_{i+2} + 8 f_{i+1} - f_i)
            if i < 1:
                continue
            # Predictor: use AB2 to predict w_pred:
            f_im1 = f_vals[i-1]
            f_i = f_vals[i]
            w_pred = w[i] + (h/2)*(3*f_i - f_im1)  # AB2 predictor for w_{i+1}
            f_pred = f_numeric(t[i+1], w_pred)
            # Now correct: w_{i+1} = w_i + (h/12)(5 f_{i+1} + 8 f_i - f_{i-1})
            w[i+1] = w[i] + (h/12)*(5*f_pred + 8*f_i - f_im1)
            log.append({
                "i": i-1,
                "t_{i-1}": t[i-1],
                "w_{i-1}": w[i-1],
                "f_{i-1}": f_im1,
                "f_i": f_i,
                "w_pred": w_pred,
                "f_pred": f_pred,
                "w_{i+1}": w[i+1]
            })
            f_vals.append(f_numeric(t[i+1], w[i+1]))

        elif order == 3:
            # AM3: w_{i+3} = w_{i+2} + (h/24)(9 f_{i+3} + 19 f_{i+2} - 5 f_{i+1} + f_i)
            if i < 2:
                continue
            f_im2 = f_vals[i-2]
            f_im1 = f_vals[i-1]
            f_i   = f_vals[i]
            # Predictor: AB3:
            w_pred = w[i] + (h/12)*(23*f_i - 16*f_im1 + 5*f_im2)
            f_pred = f_numeric(t[i+1], w_pred)
            w[i+1] = w[i] + (h/24)*(9*f_pred + 19*f_i - 5*f_im1 + f_im2)
            log.append({
                "i": i-2,
                "f_{i-2}": f_im2,
                "f_{i-1}": f_im1,
                "f_i": f_i,
                "w_pred": w_pred,
                "f_pred": f_pred,
                "w_{i+1}": w[i+1]
            })
            f_vals.append(f_numeric(t[i+1], w[i+1]))

        elif order == 4:
            # AM4: w_{i+4} = w_{i+3} + (h/720)(251 f_{i+4} + 646 f_{i+3} - 264 f_{i+2} + 106 f_{i+1} - 19 f_i)
            if i < 3:
                continue
            f_im3 = f_vals[i-3]
            f_im2 = f_vals[i-2]
            f_im1 = f_vals[i-1]
            f_i   = f_vals[i]
            # Predictor: AB4:
            w_pred = w[i] + (h/24)*(55*f_i - 59*f_im1 + 37*f_im2 - 9*f_im3)
            f_pred = f_numeric(t[i+1], w_pred)
            w[i+1] = w[i] + (h/720)*(251*f_pred + 646*f_i - 264*f_im1 + 106*f_im2 - 19*f_im3)
            log.append({
                "i": i-3,
                "f_{i-3}": f_im3,
                "f_{i-2}": f_im2,
                "f_{i-1}": f_im1,
                "f_i": f_i,
                "w_pred": w_pred,
                "f_pred": f_pred,
                "w_{i+1}": w[i+1]
            })
            f_vals.append(f_numeric(t[i+1], w[i+1]))
        else:
            raise ValueError("Adams–Moulton order must be 1..4")

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
