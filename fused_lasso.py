import numpy as np

class FusedLASSO:

	def __init__(self, optimizer='vanilla', alpha=0.01, l=0.3, momentum=0.9, n_iter=1000, verbose=False):
		optimizers = {'vanilla': self._vanilla_gd,
					'momentum' : self._momentum_gd,
					'nesterov': self._nesterov_gd}
		if optimizer not in optimizers.keys():
			raise ValueError(f"{optimizer} is not recognized. Use on of the {list(optimizers.keys())}")
		else:
			self.optimizer = optimizers[optimizer]
		self.alpha = alpha
		self.l = l
		self.momentum = momentum
		self.n_iter = n_iter
		self.verbose = verbose
		self._weights = None

	def fit(self, X):
		img_shape = X.shape
		X = X.ravel()
		self._weights = np.random.normal(size=(X.shape)) / 100
		self._velocity = np.zeros_like(self._weights)
		adjacency_matrix = self._build_adjacency_matrix(img_shape)
		for it in range(self.n_iter):
			self._weights -= self.optimizer(X, adjacency_matrix)
			if self.verbose and it > 0 and it % 100 == 0:
				print(f"{it}: MSE = {self._mse_loss(X)}")

	def fit_transform(self, X):
		self.fit(X)
		return self._weights.reshape(X.shape)


	def _build_adjacency_matrix(self, shape):
		"""
		Adjacency matrix contains indices of neighbour pixels
		"""
		x, y = shape
		a = np.zeros((x * y, 4), int)
		for i in range(x):
			for j in range(y):
				ind = i * x + j
				a[ind, 0] = ind if (i - 1) < 0 else x * (i - 1) + j
				a[ind, 1] = ind if (i + 1) >= x else x * (i + 1) + j
				a[ind, 2] = ind if (j - 1) < 0 else x * i + (j - 1)
				a[ind, 3] = ind if (j + 1) >= y else x * i + (j + 1)
		return a


	def _vanilla_gd(self, X, adj):
		return self._loss_grad(X, self._weights, adj)

	def _momentum_gd(self, X, adj):
		self._velocity = self.momentum * self._velocity
		self._velocity += self._loss_grad(X, self._weights, adj)
		return self._velocity

	def _nesterov_gd(self, X, adj):
		self._velocity = self.momentum * self._velocity
		self._velocity += self._loss_grad(X, self._weights - self.momentum * self._velocity, adj)
		return self._velocity

	def _mse_loss(self, X):
		return np.sum((X - self._weights) ** 2)

	def _mse_grad(self, X, W):
		return W - X

	def _l1_subgrad(self, W, indices):
		"""
		l1 is not differentiable, but it's convex, hence we can use subgradient
		"""
		sum = np.zeros_like(W)
		for col in range(indices.shape[1]):
			sum += np.sign(W - W[indices[:, col]])
		return sum

	def _loss_grad(self, X, W, adj):
		return self.alpha * self._mse_grad(X, W) + self.l * self._l1_subgrad(W, adj)