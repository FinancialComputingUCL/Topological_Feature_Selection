from infinite_features_selection.inf_fs import select_inf_fs
from topcorr import *


class IFS_class:
	def __init__(self, num=None, dataset_name=None, alpha=None, factor=None, step=None):
		self.num = num
		self.dataset_name = dataset_name
		self.alpha = alpha
		self.factor = factor
		self.step = step
		self.rank = None

	def fit(self, x, y=None):
		x_sel, self.rank = select_inf_fs(x, self.num, self.dataset_name, self.alpha, self.factor, self.step)
		return self

	def transform(self, x, y=None):
		return np.take(x, self.rank, axis=1)

	def fit_transform(self, x, y=None):
		self.fit(x, y)
		transformed_x = self.transform(x, y)
		return transformed_x


class TFS_class:
	def __init__(self, num=None, dataset_name=None, alpha=None, method=None, correlation_type=None, step=None):
		self.num = num
		self.dataset_name = dataset_name
		self.alpha = alpha
		self.method = method
		self.correlation_type = correlation_type
		self.rank = None
		self.step = step

	def fit(self, x, y=None):
		data = pd.DataFrame(x).fillna(method="ffill").fillna(method="bfill")
		data = data.loc[:, data.std() > 0.0]
		data = data.to_numpy()

		G = tmfg(data, self.method, self.dataset_name, self.correlation_type, self.alpha, self.step)
		centrality = nx.degree_centrality(G)
		sorted_nodes = {k: v for k, v in sorted(centrality.items(), key=lambda item: item[1], reverse=True)}
		self.rank = list(sorted_nodes.keys())[:self.num]
		return self

	def transform(self, x, y=None):
		return np.take(x, self.rank, axis=1)

	def fit_transform(self, x, y=None):
		self.fit(x, y)
		transformed_x = self.transform(x, y)
		return transformed_x


