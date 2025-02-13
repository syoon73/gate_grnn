import argparse
import numpy
import torch
from os import makedirs
from os.path import abspath, exists
from lib.data import (
	X1_DIM,
	X2_DIM,
	BASIN_LIST,
	get_dataloader,
	prepare_data,
)
from lib.model import GraphRecurNetGate
from lib.nn_fns import train_graphrnn_mse
from lib.util import add_args_str, get_argparser, save_model


### Option ###
train_fn = train_graphrnn_mse
##############


def run(
	args: argparse.Namespace,
	model_name: str,
	device: torch.device,
) -> None:
	#
	model_dir = abspath(f"./model/{model_name}/global/seed{args.seed}")

	if exists(model_dir):
		print(f"{model_dir} already exists. Skip.")
		return

	# Load data
	basin_list = BASIN_LIST
	(
		(x_train, y_train),
		_,
		_,
		_,
		adj,
	) = prepare_data(basin_list, args.data_dir, args.length, thr=args.adjthr)

	loader_train, _ = get_dataloader(
		basin_list,
		y_train,
		args.length,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.num_workers,
		pin_memory=True,
		drop_last=False,
	)

	#
	assert not exists(model_dir)
	makedirs(model_dir)
	print(f"Start: {model_dir}")

	# Create network
	network = GraphRecurNetGate(
		input_size=[X1_DIM, X2_DIM],
		output_size=1,
		**vars(args),
	)
	network.to(device)

	#
	x_train = torch.from_numpy(x_train).to(device)
	y_train = torch.from_numpy(y_train).to(device)
	adj = torch.from_numpy(adj).to(device)

	# Optimizer
	optimizer = torch.optim.RAdam(
		network.parameters(), lr=args.lr, weight_decay=args.weight_decay
	)

	# Run
	for i in range(1, args.epoch + 1):
		train_fn(x_train, y_train, adj, network, loader_train, optimizer, i)
		print()

		# Save model
		save_model(f"{model_dir}/{i}", network)

	print(f"Done {abspath(model_dir)}")


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
	run(args, model_name, device)


if __name__ == "__main__":
	main()
