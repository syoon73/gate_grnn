import argparse
import numpy
import sys
import time
import torch
from os.path import abspath, exists
from lib.data import (
	X1_DIM,
	X2_DIM,
	BASIN_LIST,
	get_dataloader,
	prepare_data,
)
from lib.metric import compute_kge, compute_nse
from lib.model import GraphRecurNetGate
from lib.nn_fns import inference_graphrnn
from lib.util import (
	add_args_str,
	compute_metric_with_cumsum_len,
	get_argparser,
	get_start_epoch,
)


### Option ###
inference_fn = inference_graphrnn
##############


def run(
	args: argparse.Namespace,
	model_name: str,
	device: torch.device,
) -> None:
	# Model dir
	model_dir = abspath(f"./model/{model_name}/global/seed{args.seed}")

	if not exists(model_dir):
		print(f"{model_dir} does not exist. Skip.")
		return

	# Log path
	log_path = f"{model_dir}/log.txt"
	start_epoch = get_start_epoch(log_path)

	if start_epoch > args.epoch:
		print(f"The validation for {model_dir} has already done. Skip.")
		return

	# Load data
	basin_list = BASIN_LIST
	(
		_,
		(x_dev, y_dev),
		_,
		_,
		adj,
	) = prepare_data(basin_list, args.data_dir, args.length, thr=args.adjthr)

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

	#
	x_dev = torch.from_numpy(x_dev).to(device)
	y_dev = torch.from_numpy(y_dev)
	adj = torch.from_numpy(adj).to(device)

	# Validation
	while start_epoch <= args.epoch:
		network_path = f"{model_dir}/{start_epoch}"

		if not exists(network_path):
			sys.stdout.write(f"\r{network_path} does not exists. Wait...\n")
			sys.stdout.flush()
			time.sleep(args.sleep)
			continue

		# Load network
		network = GraphRecurNetGate(
			input_size=[X1_DIM, X2_DIM],
			output_size=1,
			**vars(args),
		)
		network.load_state_dict(torch.load(network_path))
		network.to(device)

		# Compute
		pred_dev, target_dev = inference_fn(x_dev, y_dev, adj, network, loader_dev)

		# (# basins, 1)
		nse_dev = compute_metric_with_cumsum_len(
			compute_nse, basin_list, pred_dev, target_dev, cumsum_len_dev
		)
		kge_dev = compute_metric_with_cumsum_len(
			compute_kge, basin_list, pred_dev, target_dev, cumsum_len_dev
		)
		result_dev = numpy.hstack([nse_dev, kge_dev])

		mean_dev = numpy.nanmean(result_dev, axis=0)
		median_dev = numpy.nanmedian(result_dev, axis=0)

		# Write log
		with open(f"{model_dir}/log.txt", "a") as f:
			f.write(
				f"{start_epoch}\t{mean_dev[0]}\t{median_dev[0]}\t{mean_dev[1]}\t{median_dev[1]}\n"
			)

		print(
			f"{start_epoch}\tdev nse:\t{mean_dev[0]:.4f}\t{median_dev[0]:.4f}\t\tdev kge:\t{mean_dev[1]:.4f}\t{median_dev[1]:.4f}\t{log_path}"
		)

		start_epoch += 1

	print(f"Done {log_path}")


def main() -> None:
	# Arguments
	parser = get_argparser()
	args = parser.parse_args()

	# Setting
	numpy.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	assert torch.cuda.is_available()
	torch.backends.cudnn.deterministic = True
	device = torch.device(f"cuda:{args.gpu_id}")

	# Workspace
	model_name = add_args_str(args)

	#
	run(args, model_name, device)


if __name__ == "__main__":
	main()
