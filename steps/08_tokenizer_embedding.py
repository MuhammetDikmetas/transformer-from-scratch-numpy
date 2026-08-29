import numpy as np

rng = np.random.default_rng(0)

text = "yapay zeka muhendisligi calisiyorum"

chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(chars)


def encode(s):
    return np.array([stoi[ch] for ch in s], dtype=np.int64)


def decode(ids):
    return "".join(itos[i] for i in ids)


d_model = 8

E = rng.normal(size=(vocab_size, d_model)) * 0.1
Wl = rng.normal(size=(d_model, vocab_size)) * 0.1


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def embed_forward(ids, E):
    X = E[ids]
    return X, {"ids": ids, "vocab_size": E.shape[0]}


def embed_backward(dX, cache):
    ids, V = cache["ids"], cache["vocab_size"]
    dE = np.zeros((V, dX.shape[1]))
    np.add.at(dE, ids, dX)
    return dE


def lm_head_forward(X, Wl, targets):
    logits = X @ Wl
    p = softmax(logits)
    T = targets.shape[0]
    loss = np.mean(-np.log(p[np.arange(T), targets] + 1e-12))
    return loss, {"X": X, "Wl": Wl, "p": p, "targets": targets}


def lm_head_backward(cache):
    X, Wl, p, targets = cache["X"], cache["Wl"], cache["p"], cache["targets"]
    T = targets.shape[0]

    dlogits = p.copy()
    dlogits[np.arange(T), targets] -= 1.0
    dlogits /= T

    dWl = X.T @ dlogits
    dX = dlogits @ Wl.T
    return dX, dWl


ids = encode(text)
inputs = ids[:-1]
targets = ids[1:]

print("metin:", text)
print("sozluk boyutu:", vocab_size)
print("token id'leri:", ids[:10], "...")
print("geri cozum:", decode(ids))

X, c_emb = embed_forward(inputs, E)
loss, c_head = lm_head_forward(X, Wl, targets)

print("\nembedding cikti shape:", X.shape)
print("loss:", round(loss, 4), "| beklenen (rastgele):", round(np.log(vocab_size), 4))

dX, dWl = lm_head_backward(c_head)
dE = embed_backward(dX, c_emb)

print("\ndE:", dE.shape, "| dWl:", dWl.shape, "| dX:", dX.shape)