import numpy as np

rng = np.random.default_rng(1234)

text = ("yapay zeka muhendisligi calisiyorum ve her gun yeni seyler ogreniyorum. "
        "transformer mimarisi dikkat mekanizmasi uzerine kuruludur. "
        "dikkat mekanizmasi sayesinde model kelimeler arasindaki iliskiyi ogrenir. "
        "her token kendinden onceki tokenlara bakar ve bilgi toplar. "
        "bu sekilde model bir sonraki kelimeyi tahmin etmeyi ogrenir. ") * 8

chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(chars)
data = np.array([stoi[c] for c in text], dtype=np.int64)

block_size = 32
d_model = 64
n_head = 4
d_head = d_model // n_head
n_layer = 2
d_ff = 128


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


def rmsnorm_forward(X, g, eps=1e-6):
    rms = np.sqrt(np.mean(X ** 2, axis=-1, keepdims=True) + eps)
    xhat = X / rms
    return xhat * g, {"xhat": xhat, "rms": rms, "g": g}


def rmsnorm_backward(dout, cache):
    xhat, rms, g = cache["xhat"], cache["rms"], cache["g"]
    dg = np.sum(dout * xhat, axis=0)
    dxhat = dout * g
    dX = (dxhat - xhat * np.mean(dxhat * xhat, axis=-1, keepdims=True)) / rms
    return dX, dg


def mha_forward(X, p, cos, sin):
    T = X.shape[0]
    Q = rope_apply(split_heads(X @ p["Wq"], T), cos, sin)
    K = rope_apply(split_heads(X @ p["Wk"], T), cos, sin)
    V = split_heads(X @ p["Wv"], T)

    scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(d_head)
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = np.where(mask, -np.inf, scores)

    A = softmax(scores)
    concat = merge_heads(A @ V, T)
    out = concat @ p["Wo"]

    cache = {"X": X, "p": p, "Q": Q, "K": K, "V": V, "A": A,
             "concat": concat, "mask": mask, "cos": cos, "sin": sin, "T": T}
    return out, cache


def mha_backward(dout, cache):
    X, p, T = cache["X"], cache["p"], cache["T"]
    Q, K, V, A = cache["Q"], cache["K"], cache["V"], cache["A"]
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
    dWk = X.T @ dKm
    dWv = X.T @ dVm
    dX = dQm @ p["Wq"].T + dKm @ p["Wk"].T + dVm @ p["Wv"].T

    return dX, {"Wq": dWq, "Wk": dWk, "Wv": dWv, "Wo": dWo}


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


def block_forward(X, p, cos, sin):
    n1, c_n1 = rmsnorm_forward(X, p["g1"])
    att, c_att = mha_forward(n1, p, cos, sin)
    h = X + att

    n2, c_n2 = rmsnorm_forward(h, p["g2"])
    ff, c_ff = swiglu_forward(n2, p)
    out = h + ff

    return out, {"c_n1": c_n1, "c_att": c_att, "c_n2": c_n2, "c_ff": c_ff}


def block_backward(dout, cache):
    dn2, g_ff = swiglu_backward(dout, cache["c_ff"])
    dh_norm, dg2 = rmsnorm_backward(dn2, cache["c_n2"])
    dh = dout + dh_norm

    dn1, g_att = mha_backward(dh, cache["c_att"])
    dX_norm, dg1 = rmsnorm_backward(dn1, cache["c_n1"])
    dX = dh + dX_norm

    grads = {"g1": dg1, "g2": dg2}
    grads.update(g_att)
    grads.update(g_ff)
    return dX, grads


class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95),
                 eps=1e-8, weight_decay=0.01):
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


KEYS = ["g1", "Wq", "Wk", "Wv", "Wo", "g2", "Wg", "Wu", "Wd"]


def init_block():
    return {
        "g1": np.ones(d_model),
        "Wq": rng.normal(size=(d_model, d_model)) * 0.02,
        "Wk": rng.normal(size=(d_model, d_model)) * 0.02,
        "Wv": rng.normal(size=(d_model, d_model)) * 0.02,
        "Wo": rng.normal(size=(d_model, d_model)) * 0.02,
        "g2": np.ones(d_model),
        "Wg": rng.normal(size=(d_model, d_ff)) * 0.02,
        "Wu": rng.normal(size=(d_model, d_ff)) * 0.02,
        "Wd": rng.normal(size=(d_ff, d_model)) * 0.02,
    }


E = rng.normal(size=(vocab_size, d_model)) * 0.02
blocks = [init_block() for _ in range(n_layer)]
g_f = np.ones(d_model)
Wl = rng.normal(size=(d_model, vocab_size)) * 0.02

params = [E]
for blk in blocks:
    params += [blk[k] for k in KEYS]
params += [g_f, Wl]

cos, sin = rope_tables(block_size, d_head)


def model_forward(ids, targets):
    X = E[ids]
    caches = []
    for blk in blocks:
        X, c = block_forward(X, blk, cos, sin)
        caches.append(c)

    Xn, c_fn = rmsnorm_forward(X, g_f)
    logits = Xn @ Wl
    prob = softmax(logits)

    T = targets.shape[0]
    loss = np.mean(-np.log(prob[np.arange(T), targets] + 1e-12))

    cache = {"ids": ids, "caches": caches, "c_fn": c_fn,
             "Xn": Xn, "prob": prob, "targets": targets}
    return loss, cache


def model_backward(cache):
    ids, caches, c_fn = cache["ids"], cache["caches"], cache["c_fn"]
    Xn, prob, targets = cache["Xn"], cache["prob"], cache["targets"]
    T = targets.shape[0]

    dlogits = prob.copy()
    dlogits[np.arange(T), targets] -= 1.0
    dlogits /= T

    dWl = Xn.T @ dlogits
    dXn = dlogits @ Wl.T

    dX, dg_f = rmsnorm_backward(dXn, c_fn)

    block_grads = []
    for c in reversed(caches):
        dX, gr = block_backward(dX, c)
        block_grads.append(gr)
    block_grads.reverse()

    dE = np.zeros_like(E)
    np.add.at(dE, ids, dX)

    grads = [dE]
    for gr in block_grads:
        grads += [gr[k] for k in KEYS]
    grads += [dg_f, dWl]
    return grads


def get_batch():
    ix = rng.integers(0, len(data) - block_size - 1)
    return data[ix:ix + block_size], data[ix + 1:ix + block_size + 1]


opt = AdamW(params, lr=3e-3, weight_decay=0.01)

print("sozluk boyutu:", vocab_size)
print("toplam parametre:", sum(p.size for p in params))
print("baslangic loss beklentisi:", round(float(np.log(vocab_size)), 4))
print("\n--- egitim ---")

for step in range(3001):
    inputs, targets = get_batch()
    loss, cache = model_forward(inputs, targets)
    grads = model_backward(cache)
    opt.step(grads)

    if step % 300 == 0:
        print(f"step {step:4d}  loss {loss:.4f}")

print("\nson loss:", round(float(loss), 4))