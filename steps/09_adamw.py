import numpy as np

rng = np.random.default_rng(0)


class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
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


N, D = 64, 4

X = rng.normal(size=(N, D))
true_W = np.array([2.0, -3.0, 0.5, 1.5])
y = X @ true_W

W = np.zeros(D)
b = np.zeros(1)

opt = AdamW([W, b], lr=0.1, weight_decay=0.0)

print("hedef W:", true_W)
print("\n--- egitim ---")

for step in range(501):
    pred = X @ W + b
    diff = pred - y
    loss = np.mean(diff ** 2)

    dW = 2.0 * (X.T @ diff) / N
    db = np.array([2.0 * np.mean(diff)])

    opt.step([dW, db])

    if step % 100 == 0:
        print(f"step {step:4d}  loss {loss:.6f}")

print("\nogrenilen W:", np.round(W, 4))
print("bias:", np.round(b, 4))