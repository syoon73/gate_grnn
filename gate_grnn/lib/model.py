import math
import torch
import torch.nn as nn
from torch_geometric.nn.conv import ChebConv
from torch_geometric.nn.dense import DenseGCNConv, DenseGATConv
from typing import Any, List, Tuple


class DenseChebConv(ChebConv):
	def forward(
		self, x: torch.Tensor, adj: torch.Tensor, **kwargs: Any
	) -> torch.Tensor:
		NT, B, D = x.shape
		assert adj.shape == (B, B)

		edge_index = adj.nonzero(as_tuple=False).T
		edge_weight = adj[edge_index[0], edge_index[1]]

		# ei = torch.cat([edge_index + N * i for i in range(T)], dim=1)
		# ew = edge_weight.repeat(T).view(-1)

		x = super().forward(x, edge_index=edge_index, edge_weight=edge_weight)
		return x


class GraphRecurModule(nn.Module):
	def __init__(
		self,
		input_size: int,
		hidden_size: int,
		num_gnn_layers: int,
		num_rnn_layers: int,
		gnn_type: str = "gcn",
		rnn_type: str = "lstm",
	) -> None:
		super(GraphRecurModule, self).__init__()

		self.gnn_type = str(gnn_type)
		self.rnn_type = str(rnn_type)
		assert self.gnn_type in {"cheb", "gat", "gcn"} and self.rnn_type in {
			"gru",
			"lstm",
		}

		graph_layer = {"cheb": DenseChebConv, "gat": DenseGATConv, "gcn": DenseGCNConv}[
			self.gnn_type
		]
		recur_layer = {"gru": nn.GRU, "lstm": nn.LSTM}[self.rnn_type]

		additional_kwargs_graph = {
			"cheb": {"K": 2},
			"gat": {"heads": 2, "concat": False},
			"gcn": {},
		}[self.gnn_type]

		#
		_gnn: List = []
		for i in range(num_gnn_layers):
			if i == 0:
				n_in = input_size
			else:
				n_in = hidden_size
			n_out = hidden_size

			_gnn.append(graph_layer(n_in, n_out, **additional_kwargs_graph))
			_gnn.append(nn.GELU())
		self.gnn = nn.ModuleList(_gnn)

		self.rnn = recur_layer(
			input_size=hidden_size,
			hidden_size=hidden_size,
			num_layers=num_rnn_layers,
			bias=True,
			batch_first=False,
		)

		self.init_rnn_params()
		self.rnn.flatten_parameters()

	def init_rnn_params(self) -> None:
		# rnn
		m = self.rnn
		assert isinstance(m, (nn.GRU, nn.LSTM))
		for name, param in m.named_parameters():
			# if 'weight_ih' in name:
			# 	nn.init.xavier_normal_(param)
			if "weight_hh" in name or "weight_hr" in name:
				# fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(param)
				# gain = math.sqrt(2.0 / float(fan_in + fan_out))
				gain = 1.0 / math.sqrt(m.hidden_size)
				nn.init.orthogonal_(param, gain=gain)
			elif "bias" in name:
				if "bias_hh" in name:
					if isinstance(m, nn.LSTM):
						nn.init.zeros_(param[0])
						nn.init.ones_(param[1])
						nn.init.zeros_(param[2:])
					elif isinstance(m, nn.GRU):
						nn.init.ones_(param[0])
						nn.init.zeros_(param[1:])
					else:
						nn.init.zeros_(param)
				else:
					nn.init.zeros_(param)

	def forward(self, input: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
		# input: (N: minibatch samples, T: time, B: basins, D: dimension)
		N, T, B, D = input.shape
		assert adj.shape == (B, B)

		x = input.reshape(N * T, B, D)
		for layer in self.gnn:
			if isinstance(layer, (DenseChebConv, DenseGCNConv, DenseGATConv)):
				x = layer(x, adj, add_loop=False)
			else:
				x = layer(x)

		# (NT, B, D')	-->	(N, T, B, D')	-->	(T, N, B, D')	-->	(T, NB, D')
		x = x.view(N, T, B, -1).transpose(1, 0).reshape(T, N * B, -1)
		x, _ = self.rnn(x)
		# (T, NB, D'')	-->	(T, N, B, D'')	-->	(N, T, B, D'')
		x = x.view(T, N, B, -1).transpose(1, 0)
		assert x.shape[:-1] == (N, T, B)

		return x


class GraphRecurNetGate(nn.Module):
	def __init__(
		self,
		input_size: Tuple[int, int],
		output_size: int,
		input_proj_size: int = 32,
		hidden_size: int = 32,
		num_gnn_layers: int = 2,
		num_rnn_layers: int = 1,
		num_mlp_layers: int = 2,
		num_regression_layers: int = 1,
		gnn_type: str = "gcn",
		rnn_type: str = "lstm",
		**kwargs: Any,
	) -> None:
		super(GraphRecurNetGate, self).__init__()

		self.dense_size, self.sparse_size = input_size

		# Input projection
		if input_proj_size > 0:
			self.input_proj_dense = nn.Sequential(
				nn.Linear(self.dense_size, input_proj_size), nn.GELU()
			)
			self.input_proj_sparse = nn.Sequential(
				nn.Linear(self.sparse_size, input_proj_size), nn.GELU()
			)
		else:
			self.register_module("input_proj_dense", None)
			self.register_module("input_proj_sparse", None)

		# Dense
		n_in = input_proj_size if input_proj_size > 0 else self.dense_size
		self.dense = GraphRecurModule(
			input_size=n_in,
			hidden_size=hidden_size,
			num_gnn_layers=num_gnn_layers,
			num_rnn_layers=num_rnn_layers,
			gnn_type=gnn_type,
			rnn_type=rnn_type,
		)

		# Gate
		expansion = kwargs.get("expansion", 2)
		self.gate_dense = nn.Sequential(
			nn.Linear(hidden_size, hidden_size * expansion),
			nn.GELU(),
			nn.Linear(hidden_size * expansion, hidden_size),
			nn.Sigmoid(),
		)
		self.gate_sparse = nn.Sequential(
			nn.Linear(hidden_size, hidden_size * expansion),
			nn.GELU(),
			nn.Linear(hidden_size * expansion, hidden_size),
			nn.Sigmoid(),
		)

		# Sparse
		_sparse: List = []
		for i in range(num_mlp_layers):
			if i == 0:
				n_in = input_proj_size if input_proj_size > 0 else self.sparse_size
			else:
				n_in = hidden_size
			n_out = hidden_size

			_sparse.append(nn.Linear(n_in, n_out))
			_sparse.append(nn.GELU())
		self.sparse = nn.Sequential(*_sparse)

		# Regression
		embedding_size = hidden_size * 2
		_regression: List = [nn.BatchNorm1d(embedding_size)]

		for i in range(num_regression_layers):
			n_in = embedding_size if i == 0 else hidden_size
			n_out = output_size if i == num_regression_layers - 1 else hidden_size
			_regression.append(nn.Linear(n_in, n_out))

			if i < num_regression_layers - 1:
				_regression.append(nn.GELU())

		self.regression = nn.Sequential(*_regression)

	def forward(
		self,
		input: torch.Tensor,
		adj: torch.Tensor,
		basin_idx: torch.Tensor,
	) -> torch.Tensor:
		# input: (N, T, B, D)
		N, T, B, D = input.shape
		assert (
			adj.shape == (B, B)
			and basin_idx.shape == (N,)
			and D == self.dense_size + self.sparse_size
		)

		# Input
		x = input[..., : -self.sparse_size]
		s = torch.stack(
			[input[i, -1, b, -self.sparse_size :] for i, b in enumerate(basin_idx)],
			dim=0,
		)

		# Dense
		if self.input_proj_dense is not None:
			x = self.input_proj_dense(x)
		x = self.dense(x, adj)
		x = torch.stack([x[i, -1, b] for i, b in enumerate(basin_idx)], dim=0)

		# Sparse
		if self.input_proj_sparse is not None:
			s = self.input_proj_sparse(s)
		s = self.sparse(s)

		# Gate
		wd = self.gate_dense(x)
		ws = self.gate_sparse(x)
		x = x * wd
		s = s * ws

		# Embedding
		h = torch.cat([x, s], dim=-1)

		assert h.ndim == 2 and not torch.isnan(h).any()

		# Regression
		h = self.regression(h)

		return h
