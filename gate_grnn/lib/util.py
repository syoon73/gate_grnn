import argparse
import numpy
import time
import torch
from os import rename
from os.path import exists
from typing import Callable, List, Optional, Tuple, Union


def get_argparser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser()

	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--gpu_id", type=int, default=0)

	parser.add_argument("--data_dir", type=str, required=True)

	parser.add_argument("--gnn_type", type=str, default="gcn")
	parser.add_argument("--rnn_type", type=str, default="lstm")

	parser.add_argument("--adjthr", type=float, default=0)
	parser.add_argument("--length", type=int, default=30)

	parser.add_argument("--input_proj_size", type=int, default=32)
	parser.add_argument("--hidden_size", type=int, default=32)
	parser.add_argument("--num_gnn_layers", type=int, default=1)
	parser.add_argument("--num_rnn_layers", type=int, default=1)
	parser.add_argument("--num_mlp_layers", type=int, default=1)
	parser.add_argument("--num_regression_layers", type=int, default=1)

	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--num_workers", type=int, default=2)

	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--weight_decay", type=float, default=0)
	parser.add_argument("--epoch", type=int, default=100)
	parser.add_argument("--sleep", type=int, default=30)

	parser.add_argument("--tag", type=str, default="")

	return parser


def add_args_str(args: argparse.Namespace) -> str:
	model_name = f"{args.gnn_type}_{args.rnn_type}_adjthr_{args.adjthr}_{args.length}_hidden_size_{args.hidden_size}_num_layers_{args.num_gnn_layers}_{args.num_rnn_layers}_{args.num_mlp_layers}_num_regression_layers_{args.num_regression_layers}"

	if args.input_proj_size > 0:
		model_name += f"_input_proj_size_{args.input_proj_size}"
	if args.weight_decay > 0:
		model_name += f"_weight_decay_{args.weight_decay}"
	if args.tag != "":
		model_name += f"_{args.tag}"

	return model_name


def compute_mean_std(
	input: numpy.ndarray, axis: Optional[Union[int, Tuple[int]]] = None
) -> Tuple[numpy.ndarray, numpy.ndarray]:
	mean = numpy.nanmean(input, axis=axis, keepdims=True)
	std = numpy.nanstd(input, axis=axis, keepdims=True)
	return mean, std


def mean_std_norm(
	input: numpy.ndarray, mean: numpy.ndarray, std: numpy.ndarray
) -> numpy.ndarray:
	assert input.ndim == mean.ndim == std.ndim
	output = (input - mean) / std
	return output


def mean_std_unnorm_exp_sub(
	input: numpy.ndarray, mean: numpy.ndarray, std: numpy.ndarray, sub_val: float = 0.01
) -> numpy.ndarray:
	assert input.ndim == mean.ndim == std.ndim
	output = input * std + mean
	idx = ~numpy.isnan(output)
	output[idx] = numpy.exp(output[idx]) - sub_val
	return output


def get_subsequence(
	x: numpy.ndarray,
	y: numpy.ndarray,
	subsequence_range_str: Tuple[str, str],
	length: int,
	shift: int = 0,
	drop_front: bool = False,
	total_range_str: Tuple[str, str] = ["2005-01-01", "2023-12-31"],
) -> Tuple[numpy.ndarray, numpy.ndarray]:
	# For the labeled data only
	# x: (T, D) or (T, D, S)
	# y: (T,) or (T, S)

	total_range = [numpy.datetime64(date_str) for date_str in total_range_str]
	subsequence_range = [
		numpy.datetime64(date_str) for date_str in subsequence_range_str
	]
	assert (
		total_range[0] <= subsequence_range[0] < subsequence_range[1] <= total_range[1]
	)

	n_samples = (subsequence_range[1] - subsequence_range[0]).item().days + 1
	assert n_samples <= x.shape[0]

	i = (subsequence_range[0] - total_range[0]).item().days

	if drop_front:
		assert i >= shift
		x_sub = x[i - shift : i + n_samples - shift]
		y_sub = y[i : i + n_samples]
	else:
		assert i - length + 1 >= shift
		x_sub = x[i - length + 1 - shift : i + n_samples - shift]
		# the subsequence y[i-length+1:i] actually does not used
		y_sub = y[i - length + 1 : i + n_samples]

	assert (
		x_sub.shape[0] == y_sub.shape[0]
	)  # and not numpy.isnan(x_sub[:, :X_DIM]).any()

	return x_sub, y_sub


def save_model(model_path: str, network: torch.nn.Module) -> None:
	if exists(model_path):
		timestamp_str = time.strftime("%Y%m%d_%H%M%S")
		print(f" Already exists: {model_path}. Move it to {model_path}_{timestamp_str}")
		rename(model_path, f"{model_path}_{timestamp_str}")

	torch.save(network.state_dict(), model_path)


def read_log(path: str) -> numpy.ndarray:
	log = []

	with open(path, "rb") as f:
		for i, line in enumerate(f):
			args = line.decode().split()
			epoch = args[0]
			assert int(epoch) == i + 1
			values = [float(_) for _ in args[1:]]
			log.append(values)

	log = numpy.asarray(log, dtype=numpy.float32)

	return log


def get_start_epoch(log_path: str) -> int:
	if not exists(log_path):
		return 1
	else:
		return read_log(log_path).shape[0] + 1


def get_best_epoch(
	log_path: str, key_order: List[int] = [1, 0], max_epoch: int = 100
) -> int:
	# order = [1, 0] (individual) or [2, 3, 0, 1] (global)
	# median -> mean / test -> valid
	log = read_log(log_path)
	r, c = log.shape
	assert r >= max_epoch and c == len(key_order)
	best_idx = numpy.lexsort(tuple([log[:max_epoch, i] for i in key_order]))[-1]

	return best_idx + 1


def compute_metric_with_cumsum_len(
	metric_fn: Callable[..., numpy.ndarray],
	basin_list: List[Union[int, str]],
	pred: numpy.ndarray,
	target: numpy.ndarray,
	cumsum_len: numpy.ndarray,
	axis: int = 0,
	eps: float = 1e-02,
) -> numpy.ndarray:
	assert (
		pred.shape[0] == target.shape[0] == cumsum_len[-1] and pred.ndim == target.ndim
	)
	values = []

	for b, basin in enumerate(basin_list):
		if b == 0:
			start = 0
		else:
			start = cumsum_len[b - 1]
		end = cumsum_len[b]

		value_b = metric_fn(pred[start:end], target[start:end], axis=axis, eps=eps)

		values.append(value_b)

	values = numpy.vstack(values, dtype=numpy.float32)
	assert values.shape == (len(basin_list), 1)
	return values


def split_pred_target_by_basin(
	basin_list: List[Union[int, str]],
	cumsum_len: numpy.ndarray,
	pred: numpy.ndarray,
	target: numpy.ndarray,
) -> Tuple[dict, dict]:
	assert len(basin_list) == len(cumsum_len) and pred.shape[1] == target.shape[1]

	basin2pred = {}
	basin2target = {}

	for b, basin in enumerate(basin_list):
		start = 0 if b == 0 else cumsum_len[b - 1]
		end = cumsum_len[b]

		basin2pred[basin] = pred[:, start:end]
		basin2target[basin] = target[:, start:end]

	return basin2pred, basin2target


def split_pred_target_dateidx_by_basin(
	basin_list: List[Union[int, str]],
	cumsum_len: numpy.ndarray,
	pred: numpy.ndarray,
	target: numpy.ndarray,
	dateidx: numpy.ndarray,
) -> Tuple[dict, dict, dict]:
	assert len(basin_list) == len(cumsum_len) and pred.shape[1] == target.shape[1]

	basin2pred = {}
	basin2target = {}
	basin2dateidx = {}

	for b, basin in enumerate(basin_list):
		start = 0 if b == 0 else cumsum_len[b - 1]
		end = cumsum_len[b]

		basin2pred[basin] = pred[:, start:end]
		basin2target[basin] = target[:, start:end]
		basin2dateidx[basin] = dateidx[start:end]

	return basin2pred, basin2target, basin2dateidx
