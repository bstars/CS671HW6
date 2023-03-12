import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import matplotlib.patches as mpatches


# 7
mat = loadmat('./distilbert-base-uncased.mat')
records = mat['records']
print(np.max(records[:,2]), np.max(records[:,3]))


# 8

models = [
	"distilbert-base-uncased",
	"bert-base-uncased"
]

for i, model in enumerate(models):
	records = loadmat('./%s.mat' % (model))['records']

	val_acc = np.max(records[:,2])
	test_acc = np.max(records[:,3])
	print(val_acc, test_acc)
	plt.bar(i, val_acc, color='red', width=0.3)
	plt.bar(i + 0.3, test_acc, color='blue', width=0.3)
plt.xticks(np.arange(len(models))  + 0.2, models)

pop_a = mpatches.Patch(color='red', label='val_acc')
pop_b = mpatches.Patch(color='blue', label='test_acc')
plt.legend(handles=[pop_a, pop_b])

plt.show()

