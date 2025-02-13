import numpy
import torch
from scipy.spatial import distance_matrix
from typing import List, Tuple, Union
from .util import compute_mean_std, mean_std_norm, get_subsequence


X1_DIM = 7
X2_DIM = 10 + 1
LOG_EPS = 0.01
BASIN_LIST = list(range(9))


class IndexProvider(torch.utils.data.Dataset):
	def __init__(
		self, y: numpy.ndarray, basin_idx: int, length: int, stride: int = 1
	) -> None:
		self.basin_idx = int(basin_idx)
		self.length = int(length)
		self.stride = int(stride)

		T = y.shape[0]
		assert T == y.shape[0] and T > self.length

		is_boundary, indices = self.sliding_window_idx(T, self.length, self.stride)
		assert is_boundary
		self.indices = numpy.asarray(
			[
				# [start, end]
				numpy.arange(start, end, dtype=numpy.int32)
				for start, end in indices
				if not (numpy.isnan(y[end - 1, self.basin_idx]))
			],
			dtype=numpy.int32,
		)

		if len(self.indices) == 0:
			print("Warning: No data in this dataset.")

	def __len__(self) -> int:
		return len(self.indices)

	def __getitem__(self, index) -> Tuple[int, numpy.ndarray]:
		idx = self.indices[index]
		return self.basin_idx, idx

	@staticmethod
	def sliding_window_idx(T: int, M: int, L: int) -> Tuple[bool, numpy.ndarray]:
		if T < M:
			n_repeats = M // T
			n_remainders = M % T
			return False, numpy.hstack(
				[numpy.arange(T, dtype=numpy.int32) for _ in range(n_repeats)]
				+ [numpy.arange(n_remainders, dtype=numpy.int32)]
			)

		elif T == M:
			return True, numpy.asarray([[0, T]], dtype=numpy.int32)

		else:
			n_remainders = (T - M) % L
			start_indices = numpy.arange(
				0, T - M + L - n_remainders, L, dtype=numpy.int32
			)
			end_indices = start_indices + M
			indices = numpy.vstack([start_indices, end_indices]).T

			if n_remainders == 0:
				assert T == indices[-1, -1]
			else:
				assert 0 < n_remainders < L
				indices = numpy.vstack([indices, [T - M, T]])

			return True, indices


def prepare_data(
	basin_list: List[Union[int, str]],
	data_dir: str,
	seqlen: int,
	shift: int = 0,
	normalize_y: bool = True,
	drop_front_train: bool = True,
	thr: float = 0,
) -> Tuple[
	Tuple[numpy.ndarray, numpy.ndarray],
	Tuple[numpy.ndarray, numpy.ndarray],
	Tuple[numpy.ndarray, numpy.ndarray],
	Tuple[numpy.ndarray, numpy.ndarray],
	numpy.ndarray,
]:
	# Data
	data = numpy.stack(
		[numpy.load(f"{data_dir}/data{basin}.npy") for basin in basin_list], axis=1
	)  # (T, B, D)
	loc = numpy.asarray(
		[numpy.load(f"{data_dir}/static{basin}.npy")[:2] for basin in basin_list],
		dtype=numpy.float32,
	)  # (B, 2)

	# Adjacency matrix (based on Euclidean distance)
	adj = distance_matrix(loc, loc)
	adj = 1.0 - adj / adj.max()
	# adj[adj < numpy.quantile(numpy.tril(adj).flatten(), q)] = 0.0
	if thr > 0:
		mask = adj < thr
		adj[mask] = 0.0
		adj[~mask] = 1.0
	adj = adj.astype(numpy.float32)

	# Normalize x
	x = data[..., :-1]
	assert x.shape[-1] == X1_DIM + X2_DIM
	x[..., X1_DIM + 2 :] = numpy.log(x[..., X1_DIM + 2 :] + LOG_EPS)
	x_mean, x_std = compute_mean_std(x, axis=0)
	x = mean_std_norm(x, x_mean, x_std)
	assert x.shape[-1] == X1_DIM + X2_DIM

	# Normalize y
	y = numpy.expand_dims(data[..., -1], 2)
	if normalize_y:
		idx = ~numpy.isnan(y)
		y[idx] = numpy.log(y[idx] + LOG_EPS)
		y_mean, y_std = compute_mean_std(y, axis=0)
		y = mean_std_norm(y, y_mean, y_std)
	else:
		idx = ~numpy.isnan(y)
		log_y = y.copy()
		log_y[idx] = numpy.log(log_y[idx] + LOG_EPS)
		y_mean, y_std = compute_mean_std(log_y, axis=0)

	#
	train_range_str = ["2005-01-01", "2014-12-31"]  # 10
	dev_range_str = ["2015-01-01", "2017-12-31"]  # 3
	eval_range_str = ["2018-01-01", "2023-12-31"]  # 6

	x_train, y_train = get_subsequence(
		x,
		y,
		subsequence_range_str=train_range_str,
		length=seqlen,
		shift=shift,
		drop_front=drop_front_train,
	)
	x_dev, y_dev = get_subsequence(
		x,
		y,
		subsequence_range_str=dev_range_str,
		length=seqlen,
		shift=shift,
		drop_front=False,
	)
	x_eval, y_eval = get_subsequence(
		x,
		y,
		subsequence_range_str=eval_range_str,
		length=seqlen,
		shift=shift,
		drop_front=False,
	)

	return (x_train, y_train), (x_dev, y_dev), (x_eval, y_eval), (y_mean, y_std), adj


def get_dataloader(
	basin_list: List[Union[int, str]],
	y: numpy.ndarray,
	seqlen: int,
	batch_size: int = 32,
	shuffle: bool = False,
	num_workers: int = 1,
	pin_memory: bool = True,
	drop_last: bool = False,
) -> Tuple[torch.utils.data.DataLoader, numpy.ndarray]:
	dataset = []
	cumsum_len = []

	for b, basin in enumerate(basin_list):
		dataset_b = IndexProvider(y, b, seqlen, stride=1)
		dataset.append(dataset_b)
		cumsum_len.append(len(dataset_b))

	cumsum_len = numpy.cumsum(cumsum_len)

	loader = torch.utils.data.DataLoader(
		torch.utils.data.ConcatDataset(dataset),
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		pin_memory=pin_memory,
		drop_last=drop_last,
	)
	assert (
		len(loader) > 0 and cumsum_len[-1] > 0
	), f"Insufficient sample:\tlen: {len(loader)}\t cumsum: {cumsum_len[-1]}"

	return loader, cumsum_len
