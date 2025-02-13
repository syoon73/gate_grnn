import argparse
import numpy
import torch
from os.path import exists
from lib.data import (
	X1_DIM,
	X2_DIM,
	BASIN_LIST,
	get_dataloader,
	prepare_data,
)
from lib.model import GraphRecurNetGate
from lib.nn_fns import inference_graphrnn
from lib.util import (
	add_args_str,
	get_argparser,
	get_best_epoch,
	mean_std_unnorm_exp_sub,
	split_pred_target_by_basin,
)
from lib.metric import compute_kge, compute_nse


### Option ###
inference_fn = inference_graphrnn
##############


def run(
	args: argparse.Namespace,
	model_name: str,
	device: torch.device,
) -> numpy.ndarray:
	# Load data
	basin_list = BASIN_LIST
	(
		_,
		(x_dev, y_dev),
		(x_eval, y_eval),
		(y_mean, y_std),  # (1, B, D)
		adj,
	) = prepare_data(
		basin_list, args.data_dir, args.length, normalize_y=False, thr=args.adjthr
	)

	loader_dev, cumsum_len_dev = get_dataloader(
		basin_list,
		y_dev,
		args.length,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=True,
		drop_last=False,
	)
	loader_eval, cumsum_len_eval = get_dataloader(
		basin_list,
		y_eval,
		args.length,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=True,
		drop_last=False,
	)

	#
	x_dev = torch.from_numpy(x_dev).to(device)
	y_dev = torch.from_numpy(y_dev)
	x_eval = torch.from_numpy(x_eval).to(device)
	y_eval = torch.from_numpy(y_eval)
	adj = torch.from_numpy(adj).to(device)

	#
	pred_dev = []
	pred_eval = []

	target_dev = None
	target_eval = None

	for seed in args.seed_list:
		model_dir = f"./model/{model_name}/global/seed{seed}"
		assert exists(model_dir)

		# Model selection
		if args.last_epoch:
			best_epoch = args.epoch
		else:
			log_path = f"{model_dir}/log.txt"
			best_epoch = get_best_epoch(
				log_path, key_order=[2, 3, 0, 1], max_epoch=args.epoch
			)
		network_path = f"{model_dir}/{best_epoch}"

		# Load network
		network = GraphRecurNetGate(
			input_size=[X1_DIM, X2_DIM],
			output_size=1,
			**vars(args),
		)
		network.load_state_dict(torch.load(network_path))
		network.to(device)

		# Compute
		pred_dev_s, target_dev_s = inference_fn(x_dev, y_dev, adj, network, loader_dev)
		pred_eval_s, target_eval_s = inference_fn(
			x_eval, y_eval, adj, network, loader_eval
		)

		pred_dev.append(pred_dev_s)
		pred_eval.append(pred_eval_s)

		if target_dev is None:
			target_dev = target_dev_s
		else:
			assert numpy.array_equal(target_dev, target_dev_s)

		if target_eval is None:
			target_eval = target_eval_s
		else:
			assert numpy.array_equal(target_eval, target_eval_s)

	# (# seeds + 1, len, C=1)
	pred_dev = numpy.stack(pred_dev)
	pred_dev = numpy.vstack([pred_dev, numpy.nanmean(pred_dev, axis=0, keepdims=True)])
	pred_eval = numpy.stack(pred_eval)
	pred_eval = numpy.vstack(
		[pred_eval, numpy.nanmean(pred_eval, axis=0, keepdims=True)]
	)

	# (len, C=1) --> (1, len, C=1) --> (1, len)
	target_dev = numpy.expand_dims(target_dev, 0).squeeze(-1)
	target_eval = numpy.expand_dims(target_eval, 0).squeeze(-1)

	assert (
		pred_dev.shape[0] == pred_eval.shape[0] == len(args.seed_list) + 1
		and pred_dev.shape[1] == target_dev.shape[1]
		and pred_eval.shape[1] == target_eval.shape[1]
	)

	# Split by basin
	basin2pred_dev, basin2target_dev = split_pred_target_by_basin(
		basin_list, cumsum_len_dev, pred_dev, target_dev
	)
	basin2pred_eval, basin2target_eval = split_pred_target_by_basin(
		basin_list, cumsum_len_eval, pred_eval, target_eval
	)

	# (# basins, # seeds + 1, 4: {nse_dev}, {kge_dev}, {nse_eval}, {kge_eval})
	result = numpy.empty(
		(len(basin_list), len(args.seed_list) + 1, 4), dtype=numpy.float32
	)

	for b, basin in enumerate(basin_list):
		# (# seeds + 1, len, C=1) --> (# seeds + 1, len)
		pred_dev_b = mean_std_unnorm_exp_sub(
			basin2pred_dev[basin], y_mean[:, b : b + 1], y_std[:, b : b + 1]
		).squeeze(-1)
		pred_dev_b = numpy.maximum(pred_dev_b, 0)
		target_dev_b = basin2target_dev[basin]

		pred_eval_b = mean_std_unnorm_exp_sub(
			basin2pred_eval[basin], y_mean[:, b : b + 1], y_std[:, b : b + 1]
		).squeeze(-1)
		pred_eval_b = numpy.maximum(pred_eval_b, 0)
		target_eval_b = basin2target_eval[basin]

		nse_dev_b = compute_nse(pred_dev_b, target_dev_b, axis=1)
		kge_dev_b = compute_kge(pred_dev_b, target_dev_b, axis=1)

		nse_eval_b = compute_nse(pred_eval_b, target_eval_b, axis=1)
		kge_eval_b = compute_kge(pred_eval_b, target_eval_b, axis=1)

		result[b, :, 0] = nse_dev_b
		result[b, :, 1] = kge_dev_b
		result[b, :, 2] = nse_eval_b
		result[b, :, 3] = kge_eval_b

	return result.reshape((len(basin_list), -1))


def main() -> None:
	# Arguments
	parser = get_argparser()
	parser.add_argument("--seed_list", type=int, nargs="+", default=list(range(0, 5)))
	parser.add_argument("--last_epoch", action="store_true", default=False)
	args = parser.parse_args()

	# Setting
	assert torch.cuda.is_available()
	torch.backends.cudnn.deterministic = True
	device = torch.device(f"cuda:{args.gpu_id}")

	# Workspace
	model_name = add_args_str(args)

	#
	result = run(args, model_name, device)

	basin_list = BASIN_LIST
	for b, basin in enumerate(basin_list):
		result_b_str = "\t".join([str(f"{v:.4f}") for v in result[b]])
		print(f"{basin}\t{result_b_str}")


if __name__ == "__main__":
	main()
