import math
from torch.optim import Optimizer

class PolyLR:
	def __init__(self, optimizer: Optimizer, max_iter: int, base_lr: float, power: float = 0.9):
		self.optimizer = optimizer
		self.max_iter = max_iter
		self.base_lr = base_lr
		self.power = power
		self.iter = 0

	def step(self):
		self.iter += 1
		lr = self.base_lr * ((1.0 - self.iter / self.max_iter) ** self.power)
		for pg in self.optimizer.param_groups:
			pg["lr"] = lr
		return lr
