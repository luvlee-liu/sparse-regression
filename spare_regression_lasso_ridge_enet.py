"""
Author: LI LIU
HW2 problem1
Linear regression Download the files for this problem here. They contain a training and test
predictor matrix together with the corresponding response for three different simulated datasets.
a. Each training dataset contains 200 examples and 300 predictors. Explain whether it would
   make sense to apply least-squares regression to learn a linear model from these data and
   justify your answer briefly.
b. What error do you expect the lasso to have on the training set when lamda is very small? Why?
c. Plot the coeffcient paths for the lasso, the elastic net and ridge regression for each of the
   datasets (using only the training data). Use either scikit-learn in Python or glmnet in R.
d. Plot the training and test error of the different methods as a function of the regularization
   parameter for the three different datasets. Comment on the results, do they make sense
   considering the paths that you observed in the previous question?
e. Explain briefly how you would go about selecting the regularization parameters if you only
   had access to the training data.
f. Plot the correlation matrix (X.T)X of the predictors for each of the three training datasets.
   Does this shed any light on the paths of the different algorithms?
g. Assume that datasets 1 and 2 were generated using a linear model. Imagine that the sign of
   one of the coeffcients corresponding to the relevant predictors were flipped,
   would you be able to observe this in the data? Justify your answer briefly.
"""
"""
Author: Li Liu
Platform: Python 2.7.12 :: Anaconda 4.2.0 (64-bit)
"""
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model 
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso, ElasticNet, Ridge, lasso_path, enet_path


FIG_EXT  = '.png' #save figure as png
FIG_PATH = './'	 # save figure to fig_path
ENET_L1_RATIO = 0.2 # ElasticNet l1 ratio
"""
Part C
processing coeficients path for a dataset
using lasso elasticNet and ridge regression
"""
def dataset_path(X, y, dataset_name=''):
	print("\n\ncoefs path processing... dataset: " + dataset_name)
	
	print("\nprocessing Lasso...")
	eps = 1e-5
	alphas_lasso, coefs_lasso, _ = lasso_path(X, y, eps, fit_intercept=False)
	plot_coefs_path(dataset_name + ' Lasso coefficient paths', alphas_lasso, coefs_lasso)
	
	print("\nprocessing ElasticNet...")
	eps = 1e-5
	alphas_enet, coefs_enet, _ = enet_path(X, y, eps=eps, l1_ratio=ENET_L1_RATIO, fit_intercept=False)
	plot_coefs_path(dataset_name + ' ElasticNet coefficient paths', alphas_enet, coefs_enet)

	print("\nprocessing Ridge...")
	m_ridge = Ridge(fit_intercept=False)
	alphas = np.logspace(5, -1, 100)
	ridge_alphas, model_coef = model_path(m_ridge, X, y, alphas)
	plot_coefs_path(dataset_name +' Ridge coefficient paths', ridge_alphas, model_coef)

def model_path(model, X, y, alphas=None):
	if alphas is None:
		alphas = np.logspace(0, -4, 100)

	model_coef = []	
	for a in alphas:
		model.set_params(alpha=a)
		model.fit(X, y)
		model_coef.append(model.coef_)
	model_coef = np.transpose(model_coef)
	return alphas, model_coef	

def plot_coefs_path(title, alphas, coefs):
	# plot coefs vs regularization parameter
	plt.figure()
	ax = plt.gca()
	ax.set_xscale("log", nonposx='clip')
	colors = cycle(['b', 'r', 'g', 'c', 'k'])

	for coef_l, c in zip(coefs, colors):
	    l1 = plt.plot(alphas, coef_l, c=c)

	plt.xlabel('Regularization parameter')
	plt.ylabel('coefficients')
	plt.title(title)
	plt.axis('tight')
	plt.savefig(FIG_PATH + title+ FIG_EXT, bbox_inches='tight')


def top_n_predictors_selection(coefs, n=10):
	"""
	top n predictor selection
		select top n predicatior from coefs at alpha_index
	"""
	top_index = top_n_index(np.fabs(coefs), n)
	top_coefs = coefs[top_index]
	print 'top {0} out of {1} predictors:'.format(n , len(coefs))
	print 'predictors indexes: ', top_index 
	print 'predictors coefs:', top_coefs
	print 'sorted predictors indexes: ', np.sort(top_index)
	print 'sorted predictors coefs:', coefs[np.sort(top_index)]

def top_n_index(array, n):
	# return top n elements' index in descending order
	return np.array(array).argsort()[-n:][::-1]


"""
Part D
processing mean square error path for a dataset
using lasso elasticNet and ridge regression
"""	
def dataset_mse(X, y, Xtest, ytrue, dataset_name=''):	
	print("\n\nMSE processing... dataset: " + dataset_name)
	alphas = np.logspace(-4, 5, 100)
	
	print("\nprocessing Lasso...")
	m_lasso = Lasso(max_iter=3000, fit_intercept=False)
	_alphas, _mse0test, _ = model_mse(m_lasso, X, y, Xtest, ytrue, alphas)
	_alphas, _mse0train, _ = model_mse(m_lasso, X, y, X, y, alphas)
	
	print("\nprocessing ElasticNet...")
	m_enet = ElasticNet(max_iter=3000, fit_intercept=False, l1_ratio=ENET_L1_RATIO)
	_alphas, _mse1test, _ = model_mse(m_enet, X, y, Xtest, ytrue, alphas)
	_alphas, _mse1train, _ = model_mse(m_enet, X, y, X, y, alphas)

	print("\nprocessing Ridge...")	
	m_ridge = Ridge(fit_intercept=False)
	_alphas, _mse2test, _ = model_mse(m_ridge, X, y, Xtest, ytrue, alphas)
	_alphas, _mse2train, _ = model_mse(m_ridge, X, y, X, y, alphas)

	#plot on one figure
	plt.figure()
	ax = plt.gca()
	ax.set_xscale("log", nonposx='clip')
	ax.set_ylim(0, 1)
	
	plt.plot(alphas, _mse0train, label='Lasso training')
	plt.plot(alphas, _mse0test, label='Lasso test')
	
	plt.plot(alphas, _mse1train, label='ElasticNet training')
	plt.plot(alphas, _mse1test, label='ElasticNet test')
	
	plt.plot(alphas, _mse2train, label='Ridge training')
	plt.plot(alphas, _mse2test, label='Ridge test')

	plt.xlabel('Regularization parameter')
	plt.ylabel('relative error l2 norm')
	plt.legend(bbox_to_anchor=(1.45, 1.0), loc='upper right', borderaxespad=0.)
	title = dataset_name + ' relative error path'
	plt.title(title)
	plt.axis('tight')
	plt.savefig(FIG_PATH + title+ FIG_EXT, bbox_inches='tight')


def model_mse(model, X, y, Xtest, ytrue, alphas=None):
	"""
	fit model with X,y and test with Xtest, ytrue
	return alphas, mse, and coefs when mse is minimium
	"""
	if alphas is None:
		alphas = np.logspace(-4, 1, 100)

	model_mse = []
	for i in range(len(alphas)):
		a = alphas[i]
		model.set_params(alpha=a)
		model.fit(X, y)
		mse = 1 - r2_score(ytrue, model.predict(Xtest)) # relative error l2 norm
		model_mse.append(mse)
		if i == 0 or mse < mse_min:
			mse_min = mse
			min_mse_index = i
			min_mse_alpha = a
			min_mse_coefs = model.coef_

	print 'when alpha=10e', np.log10(min_mse_alpha)
	print "min mse: ", mse_min
	top_n_predictors_selection(min_mse_coefs, 15)
	
	return alphas, model_mse, min_mse_coefs


"""
Part F
corrlation predicator analyze
"""
def dataset_Xy_correlation(X, y, dataset_name):
	print "\ndataset_Xy_correlation processing..."
	Xcor = np.arange(len(X[0]), dtype=np.float)
	for i in range(len(X[0])):
		Xcor[i] = (np.corrcoef(X[:,i], y)[0,1])

	n = 15
	indexes = top_n_index(abs(Xcor), n)
	title = dataset_name + " correlation between predictors and y"
	print title
	print "predictors indexes", indexes
	print "predictors coefs", Xcor[indexes]
	plt.figure()
	plt.gca()
	plt.plot(Xcor)
	plt.xlabel('predictors indexes')
	plt.ylabel('correlation')
	plt.title(title)
	plt.axis('tight')
	plt.savefig(FIG_PATH + title + FIG_EXT, bbox_inches='tight')

def dataset_X_correlation(X, dataset_name):	
	print "\ndataset_X_correlation processing..."
	Xcor = np.corrcoef(X, rowvar=0)
	#Xcor = np.matmul(X.transpose(),X)
	#Xcor_filtered = np.where(abs(Xcor[:,:]) > 0.2, abs(Xcor[:,:]), 0)
	plt.figure()
	plt.gca()
	plt.imshow(Xcor)
	title = dataset_name + " correlation between predictors"
	plt.xlabel('predictors indexes')
	plt.ylabel('predictors indexes')
	plt.title(title)
	plt.axis('tight')
	plt.savefig(FIG_PATH + title + FIG_EXT, bbox_inches='tight')


"""
PART G
Datasets are generated using a linear model. 
The sign of one of the coeffcients corresponding
to the relevant predictors were flipped
"""
def one_sign_diff_fit():
	print "\none sign flipped analyse..."
	n = 200 # number of samples
	predictor_num = 300 # number of predictors
	p = 12 # number of relevant predictors
	noise = predictor_num - p # noise predictors
	m_lasso = Lasso(max_iter=3000, fit_intercept=False)
	alphas = np.logspace(-2, 1, 100)
	coefs_relevant = np.random.rand(p) * 0.1 + 0.1 
	coefs_noise = (np.random.rand(noise) - 0.5) * 0.05
	coefs = np.append(coefs_relevant, coefs_noise)
	top_indexes = top_n_index(np.fabs(coefs), p)
	print "\ncoefs: ", top_indexes
	
	partition = int(0.8 * n)
	X = (np.random.rand(n, predictor_num) - 0.5) * 2
	X = preprocess(X)
	y = np.matmul(X, coefs)

	Xtest = X[partition+1:,:]
	Xfit = X[:partition,:]
	ytrue = y[partition+1:]
	yfit = y[:partition] 

	print "\ncoefs: ", coefs[top_indexes]
	_, _, _fit_coefs = model_mse(m_lasso, Xfit, yfit, Xtest, ytrue, alphas=alphas)
	_fit_coefs_sign = np.where(_fit_coefs > 0, 1, 0)
	#dataset_mse(Xfit, yfit, Xtest, ytrue, 'dataset_')
	dataset_path(X, y, 'dataset_')


	print "\n****one sign different dataset"
	coefs_diff = coefs
	signed_diff_index = top_indexes[0]
	print "different signed coefs index:", signed_diff_index
	coefs_diff[signed_diff_index] = -coefs_diff[signed_diff_index]
	y = np.matmul(X, coefs_diff)
	ytrue = y[partition+1:]
	yfit = y[:partition] 

	# dataset_mse(Xfit, yfit, Xtest, ytrue, 'dataset_one_sign_diff')
	dataset_path(X, y, 'dataset_one_sign_diff')
	_, _, _fit_coefs_diff = model_mse(m_lasso, Xfit, yfit, Xtest, ytrue, alphas=alphas)
	_fit_coefs_diff_sign = np.where(_fit_coefs_diff > 0, 1, 0)
	print "lasso coefs index: ", top_n_index(np.fabs(_fit_coefs_diff), p)
	print "lasso coefs: ", coefs_diff[top_n_index(np.fabs(_fit_coefs_diff), p)]
	print "coefs indexes have flipped sign: ", np.nonzero(_fit_coefs_diff_sign - _fit_coefs_sign)
	exit()

def preprocess(X):
	X -= X.mean(axis=0)
	X /= X.std(axis=0)
	return X

def main():
	# run coef path and mse path correlation analysis for 3 datasets
	for i in [1, 2, 3]:
		X = np.loadtxt("X_train_" + str(i) + ".txt")
		y = np.loadtxt("y_train_" + str(i) + ".txt")
		X = preprocess(X)
		dataset_path(X, y, 'dataset_'+ str(i))
		
		Xtest = np.loadtxt("X_test_" + str(i) + ".txt")
		ytrue = np.loadtxt("y_test_" + str(i) + ".txt")
		Xtest = preprocess(Xtest)
		dataset_mse(X, y, Xtest, ytrue, 'dataset_'+ str(i))

		dataset_X_correlation(X, 'dataset_'+ str(i));

		dataset_Xy_correlation(X, y, 'dataset_'+ str(i));

	print("Done")
	print("figure saved to " + FIG_PATH)
	#plt.show()

	one_sign_diff_fit()


if __name__ == "__main__":
	main()