import numpy as np

rng = np.random.default_rng(7)

T = 16
d_model = 32
n_head = 4
d_head = d_model // n_head
d_c = 8          # MLA latent boyutu (KV sikistirma)

n_exp = 4        # yonlendirilen expert sayisi
top_k = 2        # her token kac expert secer
d_ff = 32        # her expert'in ic genisligi
bias_gamma = 0.01


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def split_heads(x, T):
    return x.reshape(T, n_head, d_head).transpose(1, 0, 2)


def merge_heads(x, T):
    return x.transpose(1, 0, 2).reshape(T, d_model)


def rope_tables(T, d, base=10000.0):
    i = np.arange(d // 2)
    theta = 1.0 / (base ** (2.0 * i / d))
    pos = np.arange(T)
    angles = pos[:, None] * theta[None, :]
    return np.cos(angles), np.sin(angles)


def rope_apply(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


def rope_apply_backward(dout, cos, sin):
    d1, d2 = dout[..., 0::2], dout[..., 1::2]
    dx = np.empty_like(dout)
    dx[..., 0::2] = d1 * cos + d2 * sin
    dx[..., 1::2] = -d1 * sin + d2 * cos
    return dx


def mla_forward(X, p, cos, sin):
    T = X.shape[0]

    c_kv = X @ p["Wdkv"]          # (T, d_c)  <- sikistirilmis latent
    K_full = c_kv @ p["Wuk"]      # (T, d_model)
    V_full = c_kv @ p["Wuv"]
    Q_full = X @ p["Wq"]

    Q = rope_apply(split_heads(Q_full, T), cos, sin)
    K = rope_apply(split_heads(K_full, T), cos, sin)
    V = split_heads(V_full, T)

    scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(d_head)
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = np.where(mask, -np.inf, scores)

    A = softmax(scores)
    concat = merge_heads(A @ V, T)
    out = concat @ p["Wo"]

    cache = {"X": X, "p": p, "c_kv": c_kv, "Q": Q, "K": K, "V": V,
             "A": A, "concat": concat, "mask": mask,
             "cos": cos, "sin": sin, "T": T}
    return out, cache


def mla_backward(dout, cache):
    X, p, T = cache["X"], cache["p"], cache["T"]
    c_kv, Q, K, V, A = (cache["c_kv"], cache["Q"], cache["K"],
                        cache["V"], cache["A"])
    concat, mask = cache["concat"], cache["mask"]
    cos, sin = cache["cos"], cache["sin"]

    dWo = concat.T @ dout
    dY = split_heads(dout @ p["Wo"].T, T)

    dA = dY @ V.transpose(0, 2, 1)
    dV = A.transpose(0, 2, 1) @ dY

    dscores = A * (dA - np.sum(dA * A, axis=-1, keepdims=True))
    dscores = np.where(mask, 0.0, dscores) / np.sqrt(d_head)

    dQ = rope_apply_backward(dscores @ K, cos, sin)
    dK = rope_apply_backward(dscores.transpose(0, 2, 1) @ Q, cos, sin)

    dQm = merge_heads(dQ, T)
    dKm = merge_heads(dK, T)
    dVm = merge_heads(dV, T)

    dWq = X.T @ dQm
    dWuk = c_kv.T @ dKm
    dWuv = c_kv.T @ dVm

    dc_kv = dKm @ p["Wuk"].T + dVm @ p["Wuv"].T
    dWdkv = X.T @ dc_kv

    dX = dQm @ p["Wq"].T + dc_kv @ p["Wdkv"].T

    return dX, {"Wq": dWq, "Wdkv": dWdkv, "Wuk": dWuk,
                "Wuv": dWuv, "Wo": dWo}


def swiglu_forward(X, p):
    gate = X @ p["Wg"]
    up = X @ p["Wu"]
    s = sigmoid(gate)
    silu = gate * s
    h = silu * up
    out = h @ p["Wd"]
    return out, {"X": X, "p": p, "gate": gate, "up": up,
                 "s": s, "silu": silu, "h": h}


def swiglu_backward(dout, cache):
    X, p = cache["X"], cache["p"]
    gate, up, s, silu, h = (cache["gate"], cache["up"], cache["s"],
                            cache["silu"], cache["h"])

    dWd = h.T @ dout
    dh = dout @ p["Wd"].T
    dsilu = dh * up
    dup = dh * silu
    dgate = dsilu * (s * (1.0 + gate * (1.0 - s)))

    dWg = X.T @ dgate
    dWu = X.T @ dup
    dX = dgate @ p["Wg"].T + dup @ p["Wu"].T

    return dX, {"Wg": dWg, "Wu": dWu, "Wd": dWd}


def moe_forward(X, p, bias):
    T = X.shape[0]

    scores = X @ p["Wr"]
    probs = softmax(scores)

    sel = probs + bias                       # bias SADECE secimde
    idx = np.argsort(sel, axis=1)[:, -top_k:]

    r = np.take_along_axis(probs, idx, axis=1)
    S = r.sum(axis=1, keepdims=True)
    w = r / S

    shared_out, c_shared = swiglu_forward(X, p["shared"])
    out = shared_out.copy()

    exp_info = []
    counts = np.zeros(n_exp)

    for e in range(n_exp):
        pos = np.argwhere(idx == e)
        if len(pos) == 0:
            exp_info.append(None)
            continue
        toks, slots = pos[:, 0], pos[:, 1]
        counts[e] = len(toks)

        y, c = swiglu_forward(X[toks], p["experts"][e])
        out[toks] += w[toks, slots][:, None] * y
        exp_info.append((toks, slots, y, c))

    cache = {"X": X, "p": p, "probs": probs, "idx": idx,
             "w": w, "S": S, "c_shared": c_shared,
             "exp_info": exp_info, "T": T}
    return out, cache, counts


def moe_backward(dout, cache):
    X, p, T = cache["X"], cache["p"], cache["T"]
    probs, idx, w, S = cache["probs"], cache["idx"], cache["w"], cache["S"]

    dX, g_shared = swiglu_backward(dout, cache["c_shared"])

    dw = np.zeros_like(w)
    exp_grads = []

    for e in range(n_exp):
        info = cache["exp_info"][e]
        if info is None:
            exp_grads.append({"Wg": np.zeros_like(p["experts"][e]["Wg"]),
                              "Wu": np.zeros_like(p["experts"][e]["Wu"]),
                              "Wd": np.zeros_like(p["experts"][e]["Wd"])})
            continue

        toks, slots, y, c = info
        dy = dout[toks] * w[toks, slots][:, None]
        dX_sub, ge = swiglu_backward(dy, c)

        np.add.at(dX, toks, dX_sub)
        dw[toks, slots] = np.sum(dout[toks] * y, axis=1)
        exp_grads.append(ge)

    dr = (dw - np.sum(dw * w, axis=1, keepdims=True)) / S

    dprobs = np.zeros_like(probs)
    np.put_along_axis(dprobs, idx, dr, axis=1)

    dscores = probs * (dprobs - np.sum(dprobs * probs, axis=1, keepdims=True))
    dWr = X.T @ dscores
    dX += dscores @ p["Wr"].T

    return dX, dWr, g_shared, exp_grads

class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95),
                 eps=1e-8, weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.t = 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1.0 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1.0 - self.b2) * (g ** 2)
            m_hat = self.m[i] / (1.0 - self.b1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.b2 ** self.t)
            p -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps))
            p -= self.lr * self.wd * p



MLA_KEYS = ["Wq", "Wdkv", "Wuk", "Wuv", "Wo"]
EXP_KEYS = ["Wg", "Wu", "Wd"]


def new_expert():
    return {"Wg": rng.normal(size=(d_model, d_ff)) * 0.05,
            "Wu": rng.normal(size=(d_model, d_ff)) * 0.05,
            "Wd": rng.normal(size=(d_ff, d_model)) * 0.05}


mla = {"Wq": rng.normal(size=(d_model, d_model)) * 0.05,
       "Wdkv": rng.normal(size=(d_model, d_c)) * 0.05,
       "Wuk": rng.normal(size=(d_c, d_model)) * 0.05,
       "Wuv": rng.normal(size=(d_c, d_model)) * 0.05,
       "Wo": rng.normal(size=(d_model, d_model)) * 0.05}

moe = {"Wr": rng.normal(size=(d_model, n_exp)) * 0.05,
       "shared": new_expert(),
       "experts": [new_expert() for _ in range(n_exp)]}

bias = np.zeros(n_exp)

params = [mla[k] for k in MLA_KEYS]
params += [moe["Wr"]]
params += [moe["shared"][k] for k in EXP_KEYS]
for e in range(n_exp):
    params += [moe["experts"][e][k] for k in EXP_KEYS]

cos, sin = rope_tables(T, d_head)

X = rng.normal(size=(T, d_model))
target = rng.normal(size=(T, d_model))

print("MLA: KV cache boyutu ->", d_c, "sayi/token")
print("     normal attention ->", 2 * d_model, "sayi/token")
print("     kazanc:", round(2 * d_model / d_c, 1), "kat")
print("\nMoE:", n_exp, "expert +", 1, "shared | token basina aktif:", top_k, "+1")
print("toplam parametre:", sum(p.size for p in params))

opt = AdamW(params, lr=5e-3)

print("\n--- egitim ---")
for step in range(801):
    h, c_mla = mla_forward(X, mla, cos, sin)
    out, c_moe, counts = moe_forward(h, moe, bias)

    diff = out - target
    loss = np.mean(diff ** 2)
    dout = 2.0 * diff / diff.size

    dh, dWr, g_shared, exp_grads = moe_backward(dout, c_moe)
    dX, g_mla = mla_backward(dh, c_mla)

    grads = [g_mla[k] for k in MLA_KEYS]
    grads += [dWr]
    grads += [g_shared[k] for k in EXP_KEYS]
    for e in range(n_exp):
        grads += [exp_grads[e][k] for k in EXP_KEYS]

    opt.step(grads)

    # aux-loss-free yuk dengeleme: bias'i yuke gore kaydir
    load = counts / (T * top_k)
    bias += bias_gamma * np.sign(1.0 / n_exp - load)

    if step % 100 == 0:
        print(f"step {step:4d}  loss {loss:.5f}  expert yuku {np.round(load, 2)}")

print("\nson loss:", round(float(loss), 5))
print("son bias :", np.round(bias, 3))