import sys
from subprocess import run


### Option ###
gpu_id = int(sys.argv[1])
length = int(sys.argv[2])
gnn_type = ["cheb", "gcn", "gat"][int(sys.argv[3])]
rnn_type = ["gru", "lstm"][int(sys.argv[4])]
adjthr = [0, 0.1, 0.2, 0.3, 0.4, 0.5][int(sys.argv[5])]

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


def train_per_seed(seed: int) -> None:
	script_name = f"train_grnn.py"
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


def main() -> None:
	seed_list = list(range(5))

	for seed in seed_list:
		train_per_seed(seed)


if __name__ == "__main__":
	main()
