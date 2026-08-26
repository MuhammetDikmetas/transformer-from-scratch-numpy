import numpy as np

rng = np.random.default_rng(0)

T, d_model, d_ff = 5, 8, 16

X = rng.normal(size=(T, d_model))
Wg = rng.normal(size=(d_model, d_ff)) * 0.1
Wu = rng.normal(size=(d_model, d_ff)) * 0.1
Wd = rng.normal(size=(d_ff, d_model)) * 0.1

G = rng.normal(size=(T, d_model))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def swiglu_forward(X, Wg, Wu, Wd):
    gate = X @ Wg
    up = X @ Wu

    s = sigmoid(gate)
    silu = gate * s
    h = silu * up

    out = h @ Wd

    cache = {"X": X, "Wg": Wg, "Wu": Wu, "Wd": Wd,
             "gate": gate, "up": up, "s": s, "silu": silu, "h": h}
    return out, cache


def swiglu_backward(dout, cache):
    X, Wg, Wu, Wd = cache["X"], cache["Wg"], cache["Wu"], cache["Wd"]
    gate, up, s, silu, h = (cache["gate"], cache["up"], cache["s"],
                            cache["silu"], cache["h"])

    dWd = h.T @ dout
    dh = dout @ Wd.T

    dsilu = dh * up
    dup = dh * silu

    dgate = dsilu * (s * (1.0 + gate * (1.0 - s)))

    dWg = X.T @ dgate
    dWu = X.T @ dup

    dX = dgate @ Wg.T + dup @ Wu.T

    return dWg, dWu, dWd, dX


out, cache = swiglu_forward(X, Wg, Wu, Wd)
print("out shape:", out.shape)
print("ara katman genisligi:", d_ff)

dWg, dWu, dWd, dX = swiglu_backward(G, cache)
print("\ndWg:", dWg.shape, "| dWu:", dWu.shape,
      "| dWd:", dWd.shape, "| dX:", dX.shape)