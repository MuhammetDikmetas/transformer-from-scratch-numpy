import numpy as np

rng = np.random.default_rng(0)

T, d_model = 5, 8
eps = 1e-6

X = rng.normal(size=(T, d_model))
g = np.ones(d_model)

G = rng.normal(size=(T, d_model))


def rmsnorm_forward(X, g, eps=1e-6):
    rms = np.sqrt(np.mean(X ** 2, axis=-1, keepdims=True) + eps)
    xhat = X / rms
    out = xhat * g

    cache = {"xhat": xhat, "rms": rms, "g": g}
    return out, cache


def rmsnorm_backward(dout, cache):
    xhat, rms, g = cache["xhat"], cache["rms"], cache["g"]

    dg = np.sum(dout * xhat, axis=0)

    dxhat = dout * g
    dX = (dxhat - xhat * np.mean(dxhat * xhat, axis=-1, keepdims=True)) / rms

    return dX, dg


out, cache = rmsnorm_forward(X, g)
print("out shape:", out.shape)
print("\nnormalizasyon oncesi satir RMS:", np.round(np.sqrt(np.mean(X ** 2, axis=-1)), 3))
print("normalizasyon sonrasi satir RMS:", np.round(np.sqrt(np.mean(out ** 2, axis=-1)), 3))

dX, dg = rmsnorm_backward(G, cache)
print("\ndX:", dX.shape, "| dg:", dg.shape)