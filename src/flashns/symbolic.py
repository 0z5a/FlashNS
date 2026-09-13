"""Independent differentiation of a generic primitive P, not the RHS itself."""

from functools import lru_cache

import sympy as s


@lru_cache(maxsize=2)
def identities(omega_rate_sign=1):
    x, y, z1, z2, theta, omega = s.symbols("x y z1 z2 Theta Omega", real=True)
    lam = s.symbols("lambda", positive=True)
    a, b, c, g1, g2 = s.symbols("a b c G1 G2", real=True)
    D = s.Matrix([[a, b], [c, -a]])
    G = s.Matrix([g1, g2])
    X, z = s.Matrix([x, y]), s.Matrix([z1, z2])
    J = s.Matrix([[0, -1], [1, 0]])
    q = z.dot(z)
    phase = lam * z.dot(X)
    primitive = s.Function("P")
    arg = s.Symbol("s", real=True)
    f = s.diff(primitive(arg), arg).subs(arg, phase)
    fp = s.diff(primitive(arg), arg, 2).subs(arg, phase)
    psi = omega * primitive(phase) / (lam**2 * q)
    temp = theta * f
    velocity = s.Matrix([-s.diff(psi, y), s.diff(psi, x)])
    curl = s.simplify(s.diff(velocity[1], x) - s.diff(velocity[0], y))
    z_t = -D.T * z
    rates = [
        z_t[0],
        z_t[1],
        -(J * z).dot(G) * omega / (lam * q),
        omega_rate_sign * lam * z1 * theta,
    ]

    def dt(expr):
        return sum(s.diff(expr, v) * r for v, r in zip([z1, z2, theta, omega], rates))

    def grad(expr):
        return s.Matrix([s.diff(expr, x), s.diff(expr, y)])

    full_velocity = D * X + velocity
    expressions = {
        "phase_transport": dt(phase) + (D * X).dot(grad(phase)),
        "divergence_from_streamfunction": s.diff(velocity[0], x)
        + s.diff(velocity[1], y),
        "curl_from_streamfunction": curl - omega * fp,
        "scalar_residual_increment": dt(temp)
        + full_velocity.dot(grad(temp))
        + velocity.dot(G),
        "vorticity_residual_increment": dt(curl)
        + full_velocity.dot(grad(curl))
        - s.diff(temp, x),
    }
    return {k: s.simplify(v) for k, v in expressions.items()}


def symbolic_report():
    return {
        "checker": "SymPy exact differentiation and simplification",
        "assumptions": "real inputs; lambda>0; zeta!=0; trace(D)=0; smooth P; F=P'",
        "scope": "local residual increments; symbolic regression, not a Lean certificate",
        "identities": {
            k: {"expression": str(v), "zero": v == 0} for k, v in identities().items()
        },
    }
