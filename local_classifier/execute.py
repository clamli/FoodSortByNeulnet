import os
import csv
import numpy as np
from neural_network import NeuralNetMLP
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

filename = "./source_data/features_data.csv"

def load_data(path):
	# features: [samples*2]
	# labels: [samples*1]
	X = []
	y = []
	with open(path, 'rb') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			X.append([int(row['taste']), int(row['health'])])
			y.append(int(row['value']))
	return np.array(X), np.array(y)




# load data
X, y = load_data(filename)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))

# plot data
colors = ['black', 'orange', 'blue', 'green', 'yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
labels = ['value 2', 'value 3', 'value 4', 'value 5', 'value 6', 'value 7', 'value 8', 'value 9']
for i in range(2, 10):
	plt.scatter(X[y==i,0],
				X[y==i,1],
				s=50,
				c=colors[i-2],
				marker='o',
				label=labels[i-2])
plt.legend()
plt.grid()
plt.show()


#### neuralNetMLP
nn = NeuralNetMLP(n_output=10,
				  n_features=X_train.shape[1],
				  n_hidden=10,
				  l2=0.1,
				  l1=0.0,
				  epochs=1000,
				  eta=0.001,
				  alpha=0.001,
				  decrease_const=0.00001,
				  shuffle=True,
				  minibaches=1,
				  random_state=1)
# training
nn.fit(X_train, y_train, print_progress=True)

# cost tendency
plt.plot(range(len(nn.cost_)), nn.cost_)
plt.ylim([1500, 5000])
plt.ylabel('Cost')
plt.xlabel('Epochs * 50')
plt.tight_layout()
plt.show()

# accuracy of training data
y_train_pred = nn.predict(X_train)
acc = float(np.sum(y_train == y_train_pred, axis=0)) / X_train.shape[0]
print('\nTraining accuracy: %.2f%%' % (acc * 100))
# accuracy of testing data
y_test_pred = nn.predict(X_test)
acc = float(np.sum(y_test == y_test_pred, axis=0)) / X_test.shape[0]
print('Testing accuracy: %.2f%%' % (acc * 100))