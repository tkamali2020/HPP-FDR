"""
neural_net.py
=============
"""

import numpy as np


def _init_weights(fan_in, fan_out, method="xavier_uniform", rng=None):
    rng = rng or np.random.default_rng()
    if method == "xavier_uniform":
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-limit, limit, size=(fan_in, fan_out))
    raise ValueError(f"Unknown weight_init method: {method}")


class AdamOptimizer:
    

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params  
        self.lr = lr
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def _relu(x):
    return np.maximum(0.0, x)


def _relu_grad(x):
    return (x > 0).astype(x.dtype)


def _softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


class MLP:
  

    def __init__(self, input_dim, output_dim, hidden_layers, output_activation,
                 weight_init="xavier_uniform", lr=1e-3, seed=None):
        rng = np.random.default_rng(seed)
        sizes = [input_dim] + list(hidden_layers) + [output_dim]
        self.n_layers = len(sizes) - 1
        self.output_activation = output_activation

        self.W = [_init_weights(sizes[i], sizes[i + 1], weight_init, rng)
                  for i in range(self.n_layers)]
        self.b = [np.zeros(sizes[i + 1]) for i in range(self.n_layers)]

        params = self.W + self.b
        self.optimizer = AdamOptimizer(params, lr=lr)

    # ------------------------------------------------------------------ #
    def forward(self, x):
        
        self._cache_a = [x]
        self._cache_z = []
        a = x
        for i in range(self.n_layers):
            z = a @ self.W[i] + self.b[i]
            self._cache_z.append(z)
            if i < self.n_layers - 1:
                a = _relu(z)
            else:
                if self.output_activation == "softmax":
                    a = _softmax(z)
                elif self.output_activation == "linear":
                    a = z
                else:
                    raise ValueError(self.output_activation)
            self._cache_a.append(a)
        return a

    # ------------------------------------------------------------------ #
    def backward(self, d_output, grad_clip_norm=None):
        
        grads_W = [None] * self.n_layers
        grads_b = [None] * self.n_layers

        delta = d_output
        for i in reversed(range(self.n_layers)):
            a_prev = self._cache_a[i]
            grads_W[i] = np.outer(a_prev, delta)
            grads_b[i] = delta
            if i > 0:
                delta = (delta @ self.W[i].T) * _relu_grad(self._cache_z[i - 1])

        grads = grads_W + grads_b
        if grad_clip_norm is not None:
            flat = np.concatenate([g.ravel() for g in grads])
            norm = np.linalg.norm(flat)
            if norm > grad_clip_norm:
                scale = grad_clip_norm / (norm + 1e-8)
                grads = [g * scale for g in grads]

        self.optimizer.step(grads)
