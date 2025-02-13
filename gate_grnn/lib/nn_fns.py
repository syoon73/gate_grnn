import numpy
import torch
from typing import Any, Tuple


def train_graphrnn_mse(
	x: torch.Tensor,
	y: torch.Tensor,
	adj: torch.Tensor,
	network: torch.nn.Module,
	loader: torch.utils.data.DataLoader,
	optimizer: torch.optim.Optimizer,
	epoch: int,
	**kwargs: Any,
) -> None:
	network.train()

	for batch_idx, (basin_idx, idx) in enumerate(loader):
		x_batch = x[idx]
		y_batch = y[idx]
		y_batch = torch.stack(
			[y_batch[i, -1, b] for i, b in enumerate(basin_idx)], dim=0
		)
		assert not torch.isnan(y_batch).any()

		optimizer.zero_grad()
		pred = network(x_batch, adj, basin_idx)
		loss = torch.nn.functional.mse_loss(pred, y_batch)
		loss.backward()

		optimizer.step()

		print(
			f"\rEpoch {epoch:3d} {numpy.float32(batch_idx+1) / numpy.float32(len(loader)) * 100:3.2f} loss {loss.tolist():.4f}",
			end="",
		)


def inference_graphrnn(
	x: torch.Tensor,
	y: torch.Tensor,
	adj: torch.Tensor,
	network: torch.nn.Module,
	loader: torch.utils.data.DataLoader,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
	network.eval()

	pred = []
	target = []

	with torch.no_grad():
		for batch_idx, (basin_idx, idx) in enumerate(loader):
			x_batch = x[idx]
			y_batch = y[idx]
			y_batch = torch.stack(
				[y_batch[i, -1, b] for i, b in enumerate(basin_idx)], dim=0
			)
			assert not torch.isnan(y_batch).any()

			logit = network(x_batch, adj, basin_idx).cpu().numpy()

			pred.append(logit)
			target.append(y_batch.numpy())

	pred = numpy.concatenate(pred, axis=0).astype(numpy.float32)
	target = numpy.concatenate(target, axis=0).astype(numpy.float32)
	assert pred.shape == target.shape

	return pred, target
