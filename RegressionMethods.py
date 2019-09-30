import random
import scipy.stats
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model, metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

# Design Matrix
def CreateDesignMatrix_X(x, y, n=5):
    '''
    Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]

		PARAMETERS
        ----------
        x : np.array
				raveled mesh of x 
        y : np.array
        		raveled mesh of y
        n : int 
        		degree of polynomial you want to fit
	'''
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2)
    X = np.ones((N,l))

    for i in range(1, n+1):
        q = int( (i) * (i+1) / 2 )
        for k in range(i+1):
            X[:,q+k] = x**(i-k) * y**k

    return X
# ---------------------------------------------------------------------------------

 # Franke function 
def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4


def Plot(self): 
	fig 	= plt.figure(figsize=(12,9))
	ax 		= fig.gca(projection='3d')
	surf 	= ax.plot_surface(self.X, self.y, self.data_noise, cmap=cm.Greens, linewidth=0, antialiased=False)
	if self.sklearn_prediction.any() != None: 
		surf = ax.plot_surface(self.X + 1, self.y, self.sklearn_prediction, cmap=cm.Oranges, linewidth=0, antialiased=False)
	if self.lehmann_prediction.any() != None: 
		surf = ax.plot_surface(self.X, self.y + 1, self.lehmann_prediction, cmap=cm.Blues, linewidth=0, antialiased=False)
		surf = ax.plot_surface(self.X + 1, self.y + 1, abs(self.lehmann_prediction - self.data_noise), cmap=cm.Reds, linewidth=0, antialiased=False)

	ax.set_zlim(-0.1, 1.2)
	fig.colorbar(surf, shrink=0.5, aspect=5)
	plt.show()
# ---------------------------------------------------------------------------------

# Class contains all the Regression Methods used for this excercise
class RegressionMethods():
	def __init__(self, function, n=100, lamda=10.0, mu=0.0, sigma=0.4, noise_factor=5, degree=2, testing_size=0.2):
		'''
		The class which handles all the logic behind the scenes.
		Has the random training data and here you can set the noise, number of points, regularization and degree.

		PARAMETERS
        ----------
        function : mathematicalFunction
				function that will be used for regression and that is computing our targets.
        n : int
        		number of datapoints to be sampled
        lamda : float 
        		regularization parameter for Lasso or Ridge Regression
        mu : float
        	 	mean value of the noise to take, default is zero mean
        sigma : float 
        		standard deviation of the noise to sample from
        noise_factor : int
        		regularizer for the noise to either make it super noisy or barely noisy
        degree : int
        		degree of the polynomial to take 
        testing_size : float
        		size of the training and test split. Value x needs to be in range 0 <= x <= 1
		'''

		# setup noise level with mean, variance and number of samples to draw
		self.n 					= n 
		self.noisy  			= np.random.normal(mu, sigma, n ) # without resampling
		self.noise 				= np.random.normal(mu, sigma, int( n * (1 - testing_size)) )
		self.degree 			= degree
		self.function 			= function # here we will only deal with the Franke Function

		self.designMatrix 		= None # design Matrix needed for variance calculation and predictions
		self.beta_OLS		    = None # need beta for the predictions
		self.beta_Ridge 		= None # likewise need beta for Ridge predictions
		# add the regularization parameter for Lasso and Ridge
		self.lamda 				= lamda

		# generate data
		self.X_raw, self.y_raw  = self.generate_data()
		self.X, self.y 			= np.meshgrid(self.X_raw, self.y_raw) # meshing for design Matrix

		self.X_train_raw, self.X_test_raw, self.y_train_raw, self.y_test_raw = self.get_train_and_test_data(self.X_raw, self.y_raw, testing_size)
		self.X_test, self.y_test   	= np.meshgrid(self.X_test_raw, self.y_test_raw)
		self.X_train, self.y_train  = np.meshgrid(self.X_train_raw, self.y_train_raw)

		# Generate Noisy Targets for the fitting process of the models
		self.train_noise = self.function(self.X_train, self.y_train) + (noise_factor *  np.random.normal(mu, sigma, self.X_train.shape ))
		self.data_noise  = self.function(self.X, self.y) + (noise_factor * np.random.normal(mu, sigma, self.X.shape))

		# TARGETS WITH NO NOISE BUT WITH AND WITHOUT SPLITS
		self.targets 			= self.function(self.X, self.y) # like self.train_noise, only without noise
		self.train_targets      = self.function(self.X_train, self.y_train) # like self.data_noise only without noise
		self.test_targets       = self.function(self.X_test, self.y_test)

		# placeholder for all the predictions throughout the models
		self.sklearn_prediction		  = None # OLS method from SKLearn with entire dataset
		self.sklearn_pred_test_train  = None # OLS method with train and test split 
		self.sklearn_ridge 			  = None # SKLearn method Ridge
		self.sklearn_lasso 			  = None # SKLearn method of Lasso (enough for this project)

		self.lehmann_prediction 	  = None # OLS from me with entire dataset and NO Noise
		self.lehmann_ridge_pred		  = None # My implementation of Ridge 

	def generate_data(self):
		X = np.linspace(-1, 3, self.n)
		y = np.exp(-X**2) + 1.5 * np.exp(-(X-2)**2) + np.random.normal(0, 0.1, X.shape)
		#X 	= np.random.rand(self.n)
		#y 	= np.random.rand(self.n)
		return X, y

	def get_train_and_test_data(self, X, y, split):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, shuffle=True)
		return X_train, X_test, y_train, y_test

	def Variance(self): 
		error = Scores(self.targets, self.lehmann_prediction)
		sigma2 = error.MeanSquaredError()
		beta_var = np.linalg.inv(self.designMatrix.T @ self.designMatrix) * sigma2
		print("Variance of Sigma Square: {}, \nVariance of Beta: {}".format(sigma2, beta_var))

	def get_confidence_interval(self, data, confidence=0.95):
		data = np.array(data)
		mean =  np.mean(data)
		standard_error = scipy.stats.sem(data)
		h = standard_error * scipy.stats.t.ppf( (1 + confidence) / 2. , len(data) - 1)
		return mean, mean - h, mean + h

	# ------- Starting with the SKLearn implementations first ---------------------
	def Sklearn_OLS(self, X, y, noise=False):
		'''
		This function predicts the test data based on the OLS implementation of the SKLearn libarary
		However, here I want to show a way of BAD PRACTICE
		We evaluate the model on the entire dataset and do not split it, so this is REALLY BAD

		PARAMETERS
        ----------
        X : np.array
				contains all the datapoints we have, comes already as np.mesh
        y : np.array
        		contains all the targets, corresponding to data X, comes as np.mesh
        noise : boolean
        		indicator of whether we should add noise to the fitting process of the model

		RETURNS
		----------
		nothing
				stores predictions straight in the member variable of the class.
		'''
		polynom 				= PolynomialFeatures(degree=self.degree)
		XY 						= polynom.fit_transform(np.array([X.ravel(), y.ravel()]).T)
		regression 				= linear_model.LinearRegression(fit_intercept=False)
		if noise: 
			regression.fit(XY, self.data_noise.reshape(-1, 1))
		else:
			regression.fit(XY, self.targets.reshape(-1, 1))
		self.sklearn_prediction = regression.predict(XY)


	def Sklearn_OLS_test_train(self, X_train, y_train, noise=False):
		'''
		This function predicts the test data based on the OLS implementation of the SKLearn libarary
		This time the model is properly evaluated on the testing dataset

		PARAMETERS
        ----------
        X_train : np.array
				contains all the datapoints, that the algorithm should train on, comes as np.mesh
        y : np.array
        		contains all the targets, corresponding to the input datapoints X_train, comes as np.mesh
        noise : boolean
        		indicator of whether we should add noise to the fitting process of the model

		RETURNS
		----------
		nothing
				stores predictions straight in the member variable of the class.
		'''
		if (X_train.shape == self.X.shape):
			print("I expected a train set of X not the entire dataset ....")
		polynom 				= PolynomialFeatures(degree=self.degree)
		XY 						= polynom.fit_transform(np.array([X_train.ravel(), y_train.ravel()]).T)
		regression 				= linear_model.LinearRegression(fit_intercept=False)
		if noise: 
			regression.fit(XY, self.train_noise.reshape(-1, 1))
		else:
			regression.fit(XY, self.train_targets.reshape(-1, 1))
		# generate the proper test values and get the score
		predict_values 			= polynom.fit_transform(np.array([self.X_test.ravel(), self.y_test.ravel()]).T)
		self.sklearn_pred_test_train = regression.predict(predict_values)


	def Sklearn_Ridge(self, lamda, X_train, y_train, noise=False):
		'''
		This function predicts the test data based on the Ridge implementation of the SKLearn libarary

		PARAMETERS
        ----------
        lamda : float 
        		regularization parameter indicating the strength of the regularization
        X_train : np.array
				contains all the datapoints, that the algorithm should train on, comes as np.mesh
        y : np.array
        		contains all the targets, corresponding to the input datapoints X_train, comes as np.mesh
        noise : boolean
        		indicator of whether we should add noise to the fitting process of the model

		RETURNS
		----------
		nothing
				stores predictions straight in the member variable of the class.
		'''
		self.lamda = lamda
		if lamda == None:
			raise ValueError("No lambda value set for Lasso regression.")
		polynom 				= PolynomialFeatures(degree=self.degree)
		XY 						= polynom.fit_transform(np.array([X_train.ravel(), y_train.ravel()]).T)
		regression 				= linear_model.Ridge(fit_intercept=True, alpha=lamda)
		if noise: 
			regression.fit(XY, self.train_noise.reshape(-1, 1))
		else:
			regression.fit(XY, self.train_targets.reshape(-1, 1))

		to_be_predicted = polynom.fit_transform(np.array([self.X_test.ravel(), self.y_test.ravel()]).T)
		#beta = regression.coef_
		#beta[0] = regression.intercept_
		self.sklearn_ridge = regression.predict(to_be_predicted)


	def Sklearn_Lasso(self, lamda, X_train , y_train, evaluate_train_error=False, noise=False):
		'''
		This function predicts the test data based on the LASSO implementation of the SKLearn libarary

		PARAMETERS
        ----------
        lamda : float 
        		regularization parameter indicating the strength of the regularization
        X_train : np.array
				contains all the datapoints, that the algorithm should train on, comes as np.mesh
        y : np.array
        		contains all the targets, corresponding to the input datapoints X_train, comes as np.mesh
        evaluate_train_error : boolean
        		indicator of whether to evaluate on train or on test data (True - eval on Train data)
        noise : boolean
        		indicator of whether we should add noise to the fitting process of the model

		RETURNS
		----------
		nothing
				stores predictions straight in the member variable of the class.
		'''
		self.lamda = lamda
		if lamda == None:
			raise ValueError("No lambda value set for Lasso regression.")
		polynom 				= PolynomialFeatures(degree=self.degree)
		XY 						= polynom.fit_transform(np.array([X_train.ravel(), y_train.ravel()]).T)
		regression 				= linear_model.Lasso(fit_intercept=True, max_iter=10000, alpha=lamda)
		if noise: 
			regression.fit(XY, self.train_noise.reshape(-1, 1))
		else:
			regression.fit(XY, self.train_targets.reshape(-1, 1))

		if evaluate_train_error: 
			to_be_predicted = polynom.fit_transform(np.array([self.X_train.ravel(), self.y_train.ravel()]).T)
		else: 
			to_be_predicted = polynom.fit_transform(np.array([self.X_test.ravel(), self.y_test.ravel()]).T)
		#beta = regression.coef_
		#beta[0] = regression.intercept_
		self.sklearn_lasso = regression.predict(to_be_predicted)



	# ------- Own implementations follow ----------------------------------------
	#def Lehmann_OLS_fit_test_train(self, X_in, y): # fitting - storing beta values
# SLIDES -------------------------- ??????????
		# # We split the data in test and training data
		# X_train, X_test, y_train, y_test = train_test_split(X, self.targets, test_size=0.2)
		# # matrix inversion to find beta
		# beta = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
		# # and then make the prediction
		# prediction = X_train.dot(beta) # not X_test ?!
# --------------------------


	def Lehmann_OLS_fit(self, X_in, y, split=False, noise=True): 
		'''
		This is the code for my own implementation of the Ordinary Least Squares
		You can change the model and have it noisy or the split by toggling the boolean values

		PARAMETERS
        ----------
        X_in : np.array
				contains all the datapoints, comes already as np.mesh
        y : np.array
        		contains all the targets, comes already as np.mesh
        split : bool 
        		boolean indicator of whether we deal with a training set or have entire dataset
        noise : bool
        	 	boolean indicator of whether we should add noise or not.

		RETURNS
		----------
		nothing
				stores beta value in the member variable of the class.
		'''
		if(X_in.shape != y.shape): 
			raise ValueError('The shape of your X and y do not match! Give me either the entire data or the training set\n Currently received: X = {}, y = {}'.format(X_in.shape, y.shape))
		
		X 					= CreateDesignMatrix_X(X_in, y, self.degree)
		self.designMatrix 	= X

		U,S,Vt 		= np.linalg.svd(X, full_matrices=True)
		S_inverse 	= np.zeros(shape=X.shape)
		S_inverse[:S.shape[0], :S.shape[0]] = np.diag(1.0 / S)

		if noise: 
			if split: 
				try: 
					self.beta_OLS = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(self.train_noise.reshape(-1, 1))
				except:
					# SVD way of beta
					# U, s, VT = np.linalg.svd(X)
					# D = np.zeros((len(U),len(VT)))
					# for i in range(0,len(VT)):
					# 	D[i,i]=s[i]
					# UT = np.transpose(U); V = np.transpose(VT); invD = np.linalg.inv(D)
					# self.beta_OLS = np.matmul(V,np.matmul(invD,UT))
					self.beta_OLS = np.dot(Vt.T, np.dot(S_inverse.T, np.dot(U.T, self.data_noise.reshape(-1, 1))))

			else:  # entire data
				try: 
					self.beta_OLS = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(self.train_noise.reshape(-1, 1))
				except:
					self.beta_OLS = np.dot(Vt.T, np.dot(S_inverse.T, np.dot(U.T, self.data_noise.reshape(-1, 1))))

		else: # no noise wanted
			if split:
				try: 
					self.beta_OLS = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(self.train_targets.reshape(-1, 1))
				except:
					self.beta_OLS = np.dot(Vt.T, np.dot(S_inverse.T, np.dot(U.T, self.train_targets.reshape(-1, 1))))
			else: # entire data
				try: 
					self.beta_OLS = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(self.targets.reshape(-1, 1))
				except: 
					self.beta_OLS = np.dot(Vt.T, np.dot(S_inverse.T, np.dot(U.T, self.targets.reshape(-1, 1))))


	def Lehmann_Ridge_fit(self, lamda, X_in, y, split=False, noise=False):
		'''
		This is the code for my own implementation of the Ridge Regression
		It basically is the OLS with the regularization parameter lambda

		PARAMETERS
        ----------
        lamda : float 
        		regularization parameter indicating the strength of the regularization
        X_in : np.array
				contains all the datapoints, comes already as np.mesh
        y : np.array
        		contains all the targets, comes already as np.mesh
        split : bool 
        		boolean indicator of whether we deal with a training set or have entire dataset
        noise : bool
        	 	boolean indicator of whether we should add noise or not.

		RETURNS
		----------
		nothing
				stores beta value in the member variable of the class.
		'''
		if(X_in.shape != y.shape): 
			raise ValueError('The shape of your X and y do not match! Give me either the entire data or the training set\n Currently received: X = {}, y = {}'.format(X_in.shape, y.shape))
		self.lamda = lamda
		if lamda == None :
			raise ValueError("No lambda value set for Ridge regression.") 

		X 					= CreateDesignMatrix_X(X_in, y, self.degree)
		self.designMatrix 	= X
		I_X 				= np.eye(np.shape(X)[1]) # idendity matrix
		if noise: 
			if split: 
				self.beta_ridge = np.linalg.inv(X.T.dot(X) + lamda * I_X).dot(X.T).dot(self.train_noise.reshape(-1,1))
			else:  # entire data
				self.beta_ridge = np.linalg.inv(X.T.dot(X) + lamda * I_X).dot(X.T).dot(self.data_noise.reshape(-1, 1))

		else: # no noise wanted
			if split:
				self.beta_ridge = np.linalg.inv(X.T.dot(X) + lamda * I_X).dot(X.T).dot(self.train_targets.reshape(-1,1))
			else: # entire data
				self.beta_ridge = np.linalg.inv(X.T.dot(X) + lamda * I_X).dot(X.T).dot(self.targets.reshape(-1,1))


	def Lehmann_Predictions(self, regression_type, designX, split=False):
		'''
		Here the final Prediction of the model happens.
		
		PARAMETERS
        ----------
        regression_type : string (in ['OLS', 'RIDGE'])
				String indicating which type of regression methods comes in either 'OLS', or 'RIDGE'
        designX : np.array - DesignMatrix
        		contains the DesignMatrix to further calculate the dot product with the betas
        		if you want to make predictions on test data, then create the Design Matrix as follows: 
        		designX_test = CreateDesignMatrix_X(X_test, y_test, poly_degree)
        		to get the proper beta parameter in the training you have to generate the design matrix with train data
        		designX_train = CreateDesignMatrix_X(X_train, y_train, poly_degree)
        split : bool 
        		boolean indicator of whether we deal with a training set or have entire dataset

		RETURNS
		----------
		nothing
				stores the predictions in the member variable of the class
		'''
		if regression_type == 'OLS':
			self.lehmann_prediction = designX.dot(self.beta_OLS)
			# if split:
			# 	try: # if some idiot feeds the train target instead of test targets
			# 		self.lehmann_prediction.reshape(self.test_targets.shape[0], self.test_targets.shape[1])
			# 	except:
			# 		print("![ATTENTION]! you are now comparing the train targets with the train predictions!!!!")
			# 		self.lehmann_prediction.reshape(self.train_targets.shape[0], self.train_targets.shape[1])
			# else: # entire data
			# 	self.lehmann_prediction.reshape(self.targets.shape[0], self.targets.shape[1])

		elif regression_type == 'RIDGE':
			self.lehmann_ridge_pred = designX.dot(self.beta_ridge)
			# if split: 
			# 	try:
			# 		self.lehmann_ridge_pred.reshape(self.test_targets.shape[0], self.test_targets.shape[1])
			# 	except: 
			# 		print("![ATTENTION]! you are now comparing the train targets with the train predictions!!!!")
			# 		self.lehmann_ridge_pred.reshape(self.train_targets.shape[0], self.train_targets.shape[1])
			# else: # entire data
			# 	self.lehmann_ridge_pred.reshape(self.targets.shape[0], self.targets.shape[1])

	
# --------------- Below here are the Score measures (MSE, R2 and Cross Validation) ------------
class Scores():
	def __init__(self):

		self.mse 	= 0
		self.r2 	= 0
		self.kfold 	= 0

	def MeanSquaredError(self, target, predicted):
		target 		= target.ravel()
		predicted 	= predicted.ravel()
		for pred, targ in zip(predicted, target):
			self.mse += (pred - targ)**2
		self.mse /= len(target) # sklearn implemented it like this somehow
		sklearn_mse = metrics.mean_squared_error(target, predicted)
		if round(sklearn_mse, 6) != round(self.mse, 6): # get a little bit of variance in the values
			print("THE MSE: {}\nYOUR MSE: {}".format(sklearn_mse, self.mse))
			self.mse =  sklearn_mse
		return self.mse


	def R2_Score(self, target, predicted):
		mean_y 		= np.mean(target.ravel())
		predicted 	= predicted.ravel()
		target 		= target.ravel()
		numerator, denominator = 0, 0
		for pred, targ in zip(predicted, target):
			numerator += (pred - targ)**2
			denominator += (targ - mean_y)**2
		self.r2 = 1 - (numerator / denominator)
		# compare with the sklearn package
		sklearn_r2 = metrics.r2_score(target, predicted)
		if round(sklearn_r2, 6) != round(self.r2, 6):
			print("THE R2: {}\nYOUR R2: {}".format(sklearn_r2, self.r2))
			self.r2 = sklearn_r2
		return self.r2


	def K_Fold_Cross_Validation(self, X, y_in, model='OLS', lamda=0.01, k_folds=4, noise=False):
		'''
		This algorithm performs the k-fold Cross Validation on the input data X and y

		PARAMETERS
		-----------
		X : np.array
				meshed numpy array that is usually used to train a classifier from the RegressionMethod class
		y : np.array
				meshed numpy array with the corresponding targets to the input data X
		model : string (default: OLS)
				string containing the model you want to perform Cross Validation on
		lamda : decimal (default: 0.01)
				float value containing the lambda value for Lasso and Ridge regression as regularization strength
		k_folds : int (default: 4)
				determining the size of the validation set and the other subsets, splits the entire data into k equally sized subsets
				where 1 subset is used for evaluating and each subset needs to be evaluated once.
		RETURNS
		-----------
		mean_error : float
				averaged Mean Squarred Error of the Test set across the k runs
		mean_r2 : float
				averaged R2 Score of the Test Dataset across the k runs
		mean_var : float
				averaged variance of the predictions across the k runs
		mean_bias : float 
				averaged bias of targets-prediction across the k runs
		'''
		# Reconstruct the non-mesh dataset and initialize scores
		dataset = np.zeros(shape=X.shape[0])
		y 	    = np.zeros(shape=y_in.shape[0])

		# get bias and variance and error
		bias   		= []
		var    		= []
		error_mse  	= []
		error_r2    = []

		for i in range(X.shape[0]): # since square matrix we only need i
			dataset[i] 	= X[i, i]
			y[i]		= y_in[i, i]
		#split the data into k equally sized bins
		sizes = dataset.shape[0] // k_folds

		# do the cross validation magic
		for train_runs in range(k_folds):
		# get the validation indices for each run (hold out data)
			start 	= train_runs * sizes
			if train_runs == (k_folds - 1): 
				end = dataset.shape[0]
			else:
				end = start + sizes
			# split data for each run in different cal and train datasets
			val_data 	= dataset[ start : end ]
			val_target  = y[ start : end ]
			# inverting the selection
			mask 			= np.ones(len(dataset), np.bool)
			mask[start:end] = 0
			train_data 		= dataset[mask]
			train_target 	= y[mask]

			# Create the Design Matrix and targets for the Training data
			train_data, train_target = np.meshgrid(train_data, train_target) # design Matrix ecpects mesh
			X 				 		 = CreateDesignMatrix_X(train_data, train_target, 5)
			if noise: 
				franke_target 		 = FrankeFunction(train_data, train_target) + (5 * np.random.normal(0, 0.5, train_data.shape ) )
			else:
				franke_target 		 = FrankeFunction(train_data, train_target) # y- values for beta

			# create Design Matrix and targets for testing data
			val_data, val_target = np.meshgrid(val_data, val_target)
			test_X 				 = CreateDesignMatrix_X(val_data, val_target, 5)

			franke_val_target 	 = FrankeFunction(val_data, val_target)

			# target = franke_val_target.ravel()
			# predicted 	= prediction.ravel()
			
			# choose the right model
			if model == 'OLS':
				beta 		= np.linalg.inv(X.T.dot(X)).dot(X.T).dot(franke_target.reshape(-1, 1))
				prediction 	= test_X.dot(beta).reshape(franke_val_target.shape[0], franke_val_target.shape[1])
				# get the MSE and R2 errors along with the variance and bias
				error_mse.append( np.sum( (franke_val_target - prediction)**2 ) / len(prediction) )
				error_r2.append(1 - np.sum( (prediction - franke_val_target)**2 ) / ( np.sum ( (franke_val_target - np.mean(franke_val_target))**2 ) ) )

				var.append(np.var(prediction))
				bias.append(np.sum( (franke_val_target - prediction) / len(prediction) ) )

			elif model == 'RIDGE':
				beta 		= np.linalg.inv(X.T.dot(X) + lamda * np.eye(np.shape(X)[1])).dot(X.T).dot(franke_target.reshape(-1, 1))
				prediction 	= test_X.dot(beta).reshape(franke_val_target.shape[0], franke_val_target.shape[1])
				# get the MSE and R2 errors along with the variance and bias
				error_mse.append( np.sum( (franke_val_target - prediction)**2 ) / len(prediction) )
				error_r2.append(1 - np.sum( (prediction - franke_val_target)**2 ) / ( np.sum ( (franke_val_target - np.mean(franke_val_target))**2 ) ) )

				var.append(np.var(prediction))
				bias.append(np.sum( (franke_val_target - prediction) / len(prediction) ) )

			elif model == 'LASSO':
				polynom 				= PolynomialFeatures(degree=5)
				XY 						= polynom.fit_transform(np.array([train_data.ravel(), franke_target.ravel()]).T)
				regression 				= linear_model.Lasso(fit_intercept=True, max_iter=10000, alpha=lamda)
				regression.fit(XY, franke_target.reshape(-1, 1))

				to_be_predicted = polynom.fit_transform(np.array([val_data.ravel(), franke_val_target.ravel()]).T)
				prediction = regression.predict(to_be_predicted)

				# get the MSE and R2 errors along with the variance and bias
				error_mse.append( np.sum( (franke_val_target.ravel() - prediction)**2 ) / len(prediction) )
				error_r2.append(1 - np.sum( (prediction - franke_val_target.ravel())**2 ) / ( np.sum ( (franke_val_target - np.mean(franke_val_target))**2 ) ) )

				var.append(np.var(prediction))
				bias.append(np.sum( (franke_val_target.ravel() - prediction) / len(prediction) ) )
				
			else: 
				raise ValueError("GOT AN UNDEFINED MODEL: {}, please use a string of ['OLS', 'RIDGE', 'LASSO'].".format(model))

		return np.mean(error_mse), np.mean(error_r2), np.mean(var), np.mean(bias)



# ------------------------------ MAIN - TESTING -----------------------------------------------------
if __name__ == "__main__":
	# Select the testing Regression Method
	testScikit 	= True # includes OLS, Ridge, LASSO
	testLehmann = True # includes OLS and Ridge own implementation
	testCV 		= True # Test Cross Validation

	lamda       = 29 # regularization parameter for Ridge and LASSO

	splitting 	= True # check NOTICE in testLehmann
	noise 	  	= True # only for both my models (OLS and Ridge)
	noise_level = 5
	k_folds     = 4 # for Cross Validation

	test 	= RegressionMethods(n=120, function=FrankeFunction, degree=5, lamda=lamda, noise_factor=noise_level)
	testing_X = CreateDesignMatrix_X(test.X_test, test.y_test, test.degree)

	if testLehmann:
		'''
		NOTICE: If you want to use the train and test split you have to  			|| NO SPLIT
		1.) enable the "splitting" variable above, 									|| disable "splitting" - splitting=False
		2.) feed the classifier with the Training data (X_train, y_train), 			|| feed with entire data (self.X, self.y)
		3.) generate the test_Design_Matrix with the testing Data (X_test, y_test)	|| use the internal design Matrix (self.designMatrix)
		4.) feed the Scores with the test targets (test_targets)	  				|| use all targets for the scores (self.targets)
		'''
		test.Lehmann_OLS_fit(test.X_train, test.y_train, split=splitting, noise=noise)
		testing_X = CreateDesignMatrix_X(test.X_test, test.y_test, test.degree)
		test.Lehmann_Predictions('OLS', testing_X, split=splitting)
		scores = Scores()
		print("\nMy OLS, split: {}, noise: {},  MSE: {}".format(splitting, noise, scores.MeanSquaredError(test.test_targets, test.lehmann_prediction)))
		print("My OLS, split: {}, noise: {},  R2 : {}".format(splitting, noise, scores.R2_Score(test.test_targets, test.lehmann_prediction)))

		test.Lehmann_Ridge_fit(test.lamda, test.X_train, test.y_train, splitting, noise)
		test.Lehmann_Predictions('RIDGE', testing_X	, split=splitting)
		scoring = Scores()
		print("\nMy Ridge, split: {}, noise: {},  MSE: {}".format(splitting, noise, scoring.MeanSquaredError(test.test_targets	, test.lehmann_ridge_pred)))
		print("My Ridge, split: {}, noise: {},  R2 : {}".format(splitting, noise, scoring.R2_Score(test.test_targets	, test.lehmann_ridge_pred)))

	if testScikit: # generates directly the predictions, don't need to fit here explicitly
		test.Sklearn_OLS(test.X, test.y, noise=noise) 
		sk_scores = Scores()
		print("\nSKLearn OLS, split: False, noise: {},  MSE: {}".format(noise, sk_scores.MeanSquaredError(test.targets, test.sklearn_prediction)))
		print("SKLearn OLS, split: False, noise: {},  R2: {}".format(noise, sk_scores.R2_Score(test.targets, test.sklearn_prediction)))

		test.Sklearn_OLS_test_train(test.X_train, test.y_train, noise=noise) # generates directly the predictions
		sk_scores = Scores()
		print("\nSKLearn OLS, split: True, noise: {},  MSE: {}".format(noise, sk_scores.MeanSquaredError(test.test_targets, test.sklearn_pred_test_train)))
		print("SKLearn OLS, split: True, noise: {},  R2: {}".format(noise, sk_scores.R2_Score(test.test_targets, test.sklearn_pred_test_train)))

		test.Sklearn_Ridge(test.lamda, test.X_train, test.y_train, noise=noise)
		sk_ridge = Scores()
		print("\nSKLearn Ridge, split: True, noise: {},  MSE: {}".format(noise, sk_ridge.MeanSquaredError(test.test_targets, test.sklearn_ridge)))
		print("SKLearn Ridge, split: True, noise: {},  R2: {}".format(noise, sk_ridge.R2_Score(test.test_targets, test.sklearn_ridge)))

		test.Sklearn_Lasso(test.lamda, test.X_train, test.y_train, evaluate_train_error=False, noise=noise)
		sk_lasso = Scores()
		print("\nSKLearn LASSO, split: True, noise: {},  MSE: {}".format(noise, sk_lasso.MeanSquaredError(test.test_targets, test.sklearn_lasso)))
		print("SKLearn LASSO, split: True, noise: {},  R2: {}".format(noise, sk_lasso.R2_Score(test.test_targets, test.sklearn_lasso)))

	if testCV:
		scor = Scores()
		#K_Fold_Cross_Validation(self, X, y_in, model='OLS', lamda=0.01, k_folds=4, noise=False)
		mse, r2, var, bias = scor.K_Fold_Cross_Validation(test.X_train, test.y_train, model='LASSO',  k_folds=k_folds, noise=noise)
		print("\nCross Validation MSE: {}\nR2 Score : {}\nVariance : {}\nBias : {}".format(mse, r2, var, bias))