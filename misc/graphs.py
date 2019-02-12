#
# Functions used for creating plots/graphs in the paper
#
#
#
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator
from sklearn import preprocessing
from math import ceil

def confusion_heatmap(conf_mat, labels):
	norm_conf = preprocessing.normalize(conf_mat, axis=1, norm='l1') * 100.0

	mask = np.ones(norm_conf.shape, dtype=bool)
	np.fill_diagonal(mask, 0)
	max_value = norm_conf[mask].max()

	fig, ax = plt.subplots(figsize=(5,5))

	cmap = mpl.cm.Purples
	norm = mpl.colors.Normalize(vmin=0, vmax=1.75 * max_value)

	res = ax.imshow(norm_conf, cmap=cmap, norm=norm,
			interpolation='nearest')

	width, height = conf_mat.shape

	cb = fig.colorbar(res)

	plt.xticks(range(width), labels, rotation='horizontal')
	plt.yticks(range(height), labels)

	plt.show()
	# plt.savefig('confusion_matrix.png', format='png')


def error_graph(conf_mat, labels):
	error = []
	totals = np.sum(conf_mat, axis=1)
	for i in range(conf_mat.shape[0]):
		error.append(1 - conf_mat[i,i] / float(totals[i]))

	error = [e * 100.0 for e in error]

	minorLocator = AutoMinorLocator(5)

	# plt.style.use('seaborn-deep')
	fig, ax = plt.subplots(figsize=(7,5))

	prop_cycle = plt.rcParams['axes.prop_cycle']
	colors = prop_cycle.by_key()['color']

	ind = np.arange(len(error))
	width = 0.35
	colors = colors[:conf_mat.shape[0]]

	p1 = ax.bar(ind, error, width, color=colors, tick_label=labels)

	ax.yaxis.set_minor_locator(minorLocator)

	ax.set_ylabel('Percent Error', fontsize=16)
	ax.set_yticks(np.arange(0, ceil(max(error)), 0.5))
	plt.legend(p1, labels)

	plt.show()


# def error_graph_per_class(conf_mat, labels):
# 	tot_error = []
# 	totals = np.sum(conf_mat, axis=1)
# 	for i in range(conf_mat.shape[0]):
# 		# error will be in format [group, column, error]
# 		error = []
# 		for j in range(conf_mat.shape[1]):
# 			if i == j:
# 				error.append(0)
# 			else:
# 				error.append((conf_mat[i,j] / float(totals[i])) * 100.0)
# 		tot_error.append(error)

# 	minorLocator = AutoMinorLocator(5)

# 	# plt.style.use('seaborn-deep')
# 	fig, ax = plt.subplots(figsize=(7,5))

# 	prop_cycle = plt.rcParams['axes.prop_cycle']
# 	colors = prop_cycle.by_key()['color']

# 	ind = np.arange(len(error))
# 	width = 0.35
# 	colors = colors[:conf_mat.shape[0]]

# 	p1 = ax.bar(ind, error, width, color=colors, tick_label=labels)

# 	ax.yaxis.set_minor_locator(minorLocator)

# 	ax.set_ylabel('Percent Error', fontsize=16)
# 	ax.set_yticks(np.arange(0, ceil(max(error)), 0.5))
# 	plt.legend(p1, labels)

# 	plt.show()


conf_mat = np.asarray([[119144, 60, 394, 427],
					[352, 134432, 43, 218],
					[583, 38, 102495, 1919],
					[9, 28, 147, 44831]])  

labels = ["liquid", "fcc", "hcp", "bcc"]  

error_graph(conf_mat, labels)

confusion_heatmap(conf_mat, labels)




















