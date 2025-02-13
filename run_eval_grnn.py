import sys
from os import makedirs
from os.path import dirname, exists
from subprocess import run
from typing import List


### Option ###
gpu_id = int(sys.argv[1])
length = int(sys.argv[2])
gnn_type = ["cheb", "gcn", "gat"][int(sys.argv[3])]
rnn_type = ["gru", "lstm"][int(sys.argv[4])]
adjthr = [0, 0.1, 0.2, 0.3, 0.4, 0.5][int(sys.argv[5])]
last_epoch = False

python_str = "python3"

args = {
	"gpu_id": gpu_id,
	"data_dir": "/media/syoon/hydro/data/Ecology",
	"gnn_type": gnn_type,
	"rnn_type": rnn_type,
	"adjthr": adjthr,
	"length": length,
	"input_proj_size": 16,
	"hidden_size": 16,
	"num_gnn_layers": 2,
	"num_rnn_layers": 2,
	"num_mlp_layers": 2,
	"num_regression_layers": 2,
	"batch_size": 32,
	"num_workers": 1,
	"lr": 5e-04,
	"weight_decay": 1e-05,
	"epoch": 100,
}
##############


def valid_per_seed(seed: int) -> None:
	script_name = f"valid_grnn.py"
	args_str = " ".join(
		f"--{k}" if isinstance(v, bool) and v else f"--{k} {v}"
		for k, v in args.items()
		if not (isinstance(v, bool) and (not v))
	)

	cmd = f"{python_str} {script_name} {args_str} --seed {seed}"

	for additional_args_str in [
		"",
	]:
		run(f"{cmd} {additional_args_str}", cwd=f"./gate_grnn", shell=True)


def ensemble(seed_list: List[int], last_epoch: bool = False) -> None:
	script_name = f"ensemble_grnn.py"
	args_str = " ".join(
		f"--{k}" if isinstance(v, bool) and v else f"--{k} {v}"
		for k, v in args.items()
		if not (isinstance(v, bool) and (not v))
	)
	seed_list_str = f'--seed_list {" ".join([str(_) for _ in seed_list])}'

	cmd = (
		f'{python_str} {script_name} {args_str}{" --last_epoch" if last_epoch else ""}'
	)

	for fname_tag, additional_args_str in zip(
		[
			f"_hidden_size_{args['hidden_size']}_layers_{args['num_gnn_layers']}_{args['num_rnn_layers']}_{args['num_mlp_layers']}_{args['num_regression_layers']}",
		],
		[
			"",
		],
	):
		log_path = f'./log{"_last_epoch" if last_epoch else ""}/{args["gnn_type"]}_{args["rnn_type"]}_{args["adjthr"]}_{length}{fname_tag}_global.txt'

		if not exists(log_path):
			log_dir = dirname(log_path)
			if not exists(log_dir):
				makedirs(log_dir)

			run(
				f'{cmd} {additional_args_str} {seed_list_str} > {f"../{log_path}"}',
				cwd=f"./gate_grnn",
				shell=True,
			)
		else:
			print(f"{log_path} already exists. Skip.")


def main() -> None:
	seed_list = list(range(5))

	for seed in seed_list:
		valid_per_seed(seed)

	ensemble(seed_list, last_epoch=last_epoch)


if __name__ == "__main__":
	main()
