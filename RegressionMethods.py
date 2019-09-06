import random
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model, metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

# Design Matrix
def CreateDesignMatrix_X(x, y, n=5):
    """
    Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
    Input is x and y mesh or raveled mesh, keyword agruments n is the degree of the polynomial you want to fit.
    """
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2)
    X = np.ones((N,l))

    for i in range(1,n+1):
        q = int((i)*(i+1)/2)
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
# ---------------------------------------------------------------------------------

# Class contains all the Regression Methods used for this excercise
class RegressionMethods():
	def __init__(self, function, n=100, lamda=10, mu=0.0, sigma=0.4, noise_factor=5, degree=2, testing_size=0.2):
		'''
		The class which handles all the logic behind the scenes.
		Has the random training data and here you can set the noise, number of points, regularization and degree.

		PARAMETERS
        ----------
        function : mathematicalFunction
				function that will be used for regression and that is computing our targets.
        n : int
        		number of datapoints to be sampled
        lamda : int 
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

		self.XY 				= None # design Matrix needed for variance calculation and predictions
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

		# Setting up our targets WITH NOISE AND SPLIT
		try:
			self.z 					= self.function(self.X_train, self.y_train) + self.noise
		except: 
			print("Shape mismatch of noise {} and training split {}, but fixxin it right away".format(self.noise.shape, self.X_train.shape))
			self.noise 				= np.random.normal(mu, sigma, self.X_train.shape )
			self.z 					= self.function(self.X_train, self.y_train) + (noise_factor * self.noise)
			pass
		
		# NOISY TARGETS WITHOUT SPLIT OF THE DATA
		try:
			self.z_noise			= self.function(self.X, self.y) + (noise_factor * self.noisy)
		except: 
			self.noisy 				= np.random.normal(mu, sigma, self.X.shape )
			self.z_noise			= self.function(self.X, self.y) + (noise_factor * self.noisy)
			pass
		# TARGETS WITH NO NOISE BUT WITH AND WITHOUT SPLITS
		self.targets 			= self.function(self.X, self.y) # like self.z_noise, only without noise
		self.train_targets      = self.function(self.X_train, self.y_train) # like self.z only without noise
		self.test_targets       = self.function(self.X_test, self.y_test)

		# placeholder for all the predictions throughout the models
		self.sklearn_prediction		  = None # OLS method from SKLearn with entire dataset
		self.sklearn_pred_test_train  = None # OLS method with train and test split 
		self.sklearn_ridge 			  = None # SKLearn method Ridge
		self.sklearn_lasso 			  = None # SKLearn method of Lasso (enough for this project)

		self.lehmann_prediction 	  = None # OLS from me with entire dataset and NO Noise
		self.lehmann_ridge 			  = None # My implementation of Ridge 

	def generate_data(self):
		X 	= np.random.rand(self.n)
		y 	= np.random.rand(self.n)
		return X, y

	def get_train_and_test_data(self, X, y, split):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, shuffle=True)
		return X_train, X_test, y_train, y_test

	# ------- Starting with the SKLearn implementations first ---------------------
	def Sklearn_OLS(self, X, y, with_split=False): # without train and test split - can go straight to prediction
		polynom 				= PolynomialFeatures(degree=self.degree)
		XY 						= polynom.fit_transform(np.array([X.ravel(), y.ravel()]).T)
		regression 				= linear_model.LinearRegression(fit_intercept=False)
		regression.fit(XY, self.z_noise.reshape(-1, 1))
		self.sklearn_prediction = regression.predict(XY)
		self.sklearn_prediction = self.sklearn_prediction.reshape(self.z_noise.shape[0], self.z_noise.shape[1])


	def Sklearn_Ridge(self, lamda, X_in, y): # can go straight to prediction
		self.lamda = lamda
		if lamda == None :
			raise ValueError("No lambda value set for Ridge regression.")
		polynom 				= PolynomialFeatures(degree=self.degree)
		XY 						= polynom.fit_transform(np.array([X_in.ravel(), y.ravel()]).T)
		regression 				= linear_model.Ridge(fit_intercept=True, alpha=self.lamda)
		regression.fit(XY, self.z_noise.reshape(-1, 1))
		#beta = regression.coef_
		#beta[0] = regression.intercept_
		self.sklearn_ridge = regression.predict(XY)
		self.sklearn_ridge = self.sklearn_ridge.reshape(self.z_noise.shape[0], self.z_noise.shape[1])


	def Sklearn_Lasso(self, lamda, X_in , y): # can go straight to prediction
		self.lamda = lamda
		if lamda == None:
			raise ValueError("No lambda value set for Lasso regression.")
		polynom 				= PolynomialFeatures(degree=self.degree)
		XY 						= polynom.fit_transform(np.array([X_in.ravel(), y.ravel()]).T)
		regression 				= linear_model.Lasso(fit_intercept=True, max_iter=10000, alpha=lamda)
		regression.fit(XY, self.z_noise.reshape(-1, 1))
		#beta = regression.coef_
		#beta[0] = regression.intercept_
		self.sklearn_lasso = regression.predict(XY)
		self.sklearn_lasso = self.sklearn_lasso.reshape(self.z_noise.shape[0], self.z_noise.shape[1])


	def Sklearn_OLS_test_train(self, X_train, y_train): # can go straight to prediction
		if (X_train.shape == self.X.shape):
			print("I expected a train set of X not the entire dataset ....")
		polynom 				= PolynomialFeatures(degree=self.degree)
		XY 						= polynom.fit_transform(np.array([X_train.ravel(), y_train.ravel()]).T)
		regression 				= linear_model.LinearRegression(fit_intercept=False)
		regression.fit(XY, self.z.reshape(-1, 1))
		predict_values 			= polynom.fit_transform(np.array([self.X_test.ravel(), self.y_test.ravel()]).T)
		self.sklearn_pred_test_train = regression.predict(predict_values)
		self.sklearn_pred_test_train = self.sklearn_prediction.reshape(self.n, self.n)

	# ------- Own implementations follow ----------------------------------------
	def Lehmann_OLS_test_train(self, X_in, y): # fitting - storing beta values
		# creating the OLS according to the lecture slides
		X 		= CreateDesignMatrix_X(X_in, y, self.degree)
		self.XY = X
		# more stable SVD version
		U,S,Vt = np.linalg.svd(X, full_matrices=True)
		S_inverse = np.zeros(shape=X.shape)
		S_inverse[:S.shape[0], :S.shape[0]] = np.diag(1.0 / S)
		self.beta_OLS = np.dot(Vt.T, np.dot(S_inverse.T, np.dot(U.T, self.z.reshape(-1, 1))))
		# prediction - change this to test
		# self.lehmann_pred_test_train = np.dot(self.X, self.beta).reshape(self.z.shape[0], self.z.shape[1])
# SLIDES -------------------------- ??????????
		# # We split the data in test and training data
		# X_train, X_test, y_train, y_test = train_test_split(X, self.targets, test_size=0.2)
		# # matrix inversion to find beta
		# beta = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
		# # and then make the prediction
		# prediction = X_train.dot(beta) # not X_test ?!
# --------------------------
		# unstable Matrix inversion
		# X 		= CreateDesignMatrix_X(self.X_train, self.y_train, self.degree)
		# self.XY = X
		# beta 	= np.linalg.inv(X.T.dot(X)).dot(X.T).dot(self.z.reshape(-1, 1)) # is it self.targets or self.y_train
		# prediction = X.dot(beta)
		# self.lehmann_pred_test_train = prediction.reshape(self.z.shape[0], self.z.shape[1])


	def Lehmann_OLS(self, X_in, y, with_split=False, with_noise=True): 
		'''
		This is the code for my own implementation of the Ordinary Least Squares
		You can change the model and have it noisy or the split by toggling the boolean values

		PARAMETERS
        ----------
        X_in : np.array
				contains all the datapoints, comes already as np.mesh
        y : np.array
        		contains all the targets, comes already as np.mesh
        with_split : bool 
        		boolean indicator of whether we deal with a training set or have entire dataset
        with_noise : bool
        	 	boolean indicator of whether we should add noise or not.

		RETURNS
		----------
		nothing
				stores beta value in the member variable of the class.
		'''
		if(X_in.shape != y.shape): 
			raise ValueError('The shape of your X and y do not match! Give me either the entire data or the training set\n Currently received: X = {}, y = {}'.format(X_in.shape, y.shape))
		
		X 			= CreateDesignMatrix_X(X_in, y, self.degree)
		self.XY 	= X
		U,S,Vt 		= np.linalg.svd(X, full_matrices=True)
		S_inverse 	= np.zeros(shape=X.shape)
		S_inverse[:S.shape[0], :S.shape[0]] = np.diag(1.0 / S)

		if with_noise: 
			noisy 	= np.random.normal(0, 0.4, X_in.shape[0])
			z_noisy	= self.function(X_in, y) + (5 * noisy)

			if with_split: 
				try:
					self.beta_OLS = np.dot(Vt.T, np.dot(S_inverse.T, np.dot(U.T, z_noisy.reshape(-1, 1))))
				except Exception as e:
					print("\nDimensions again: {}".format(e))
					self.beta_OLS = np.dot(Vt.T, np.dot(S_inverse.T, np.dot(U.T, self.z.reshape(-1, 1))))
			else:  # entire data
				self.beta_OLS = np.dot(Vt.T, np.dot(S_inverse.T, np.dot(U.T, self.z_noise.reshape(-1, 1))))

		else: # no noise wanted
			if with_split:
				self.beta_OLS = np.dot(Vt.T, np.dot(S_inverse.T, np.dot(U.T, self.train_targets.reshape(-1, 1))))
			else: # entire data
				self.beta_OLS = np.dot(Vt.T, np.dot(S_inverse.T, np.dot(U.T, self.targets.reshape(-1, 1))))


	def Lehmann_Ridge(self, lamda, X_in, y, with_split=False, with_noise=False):
		'''
		This is the code for my own implementation of the Ordinary Least Squares
		You can change the model and have it noisy or the split by toggling the boolean values

		PARAMETERS
        ----------
        lamda : float 
        		regularization parameter indicating the strength of the regularization
        X_in : np.array
				contains all the datapoints, comes already as np.mesh
        y : np.array
        		contains all the targets, comes already as np.mesh
        with_split : bool 
        		boolean indicator of whether we deal with a training set or have entire dataset
        with_noise : bool
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

		X 			= CreateDesignMatrix_X(X_in, y, self.degree)
		self.XY 	= X
		I_X 		= np.eye(np.shape(X)[1]) # idendity matrix
		if with_noise: 
			noisy 	= np.random.normal(0, 0.4, X_in.shape[0])
			z_noisy	= self.function(X_in, y) + (5 * noisy)

			if with_split: 
				try:
					self.beta_ridge = np.linalg.inv(X.T.dot(X) + lamda * I_X).dot(X.T).dot(z_noisy.reshape(-1,1))
				except Exception as e:
					print("\nDimensions again: {}".format(e))
					self.beta_ridge = np.linalg.inv(X.T.dot(X) + lamda * I_X).dot(X.T).dot(self.z.reshape(-1, 1))
			else:  # entire data
				self.beta_ridge = np.linalg.inv(X.T.dot(X) + lamda * I_X).dot(X.T).dot(self.z_noise.reshape(-1, 1))

		else: # no noise wanted
			if with_split:
				self.beta_ridge = np.linalg.inv(X.T.dot(X) + lamda * I_X).dot(X.T).dot(self.train_targets.reshape(-1,1))
			else: # entire data
				self.beta_ridge = np.linalg.inv(X.T.dot(X) + lamda * I_X).dot(X.T).dot(self.targets.reshape(-1,1))


	def Lehmann_Predictions(self, regression_type, X_in, with_split=False):
		'''
		Here the final Prediction of the model happens.
		
		PARAMETERS
        ----------
        regression_type : string (in ['OLS', 'RIDGE'])
				String indicating which type of regression methods comes in either 'OLS', or 'RIDGE'
        X_in : np.array - DesignMatrix
        		contains the DesignMatrix to further calculate the dot product with the betas
        with_split : bool 
        		boolean indicator of whether we deal with a training set or have entire dataset

		RETURNS
		----------
		nothing
				stores the predictions in the member variable of the class
		'''
		if regression_type == 'OLS':
			self.lehmann_prediction = self.XY.dot(self.beta_OLS)
			if with_split:
				self.lehmann_prediction.reshape(self.train_targets.shape[0], self.train_targets.shape[1])
			else: # entire data
				self.lehmann_prediction.reshape(self.targets.shape[0], self.targets.shape[1])

		elif regression_type == 'RIDGE':
			self.lehmann_ridge = self.XY.dot(self.beta_ridge)
			if with_split: 
				self.lehmann_ridge.reshape(self.train_targets.shape[0], self.train_targets.shape[1])
			else: # entire data
				self.lehmann_ridge.reshape(self.targets.shape[0], self.targets.shape[1])


	def Plot(self): 
		fig 	= plt.figure(figsize=(12,9))
		ax 		= fig.gca(projection='3d')
		surf 	= ax.plot_surface(self.X, self.y, self.z_noise, cmap=cm.Greens, linewidth=0, antialiased=False)
		if self.sklearn_prediction.any() != None: 
			surf = ax.plot_surface(self.X + 1, self.y, self.sklearn_prediction, cmap=cm.Oranges, linewidth=0, antialiased=False)
		if self.lehmann_prediction.any() != None: 
			surf = ax.plot_surface(self.X, self.y + 1, self.lehmann_prediction, cmap=cm.Blues, linewidth=0, antialiased=False)
			surf = ax.plot_surface(self.X + 1, self.y + 1, abs(self.lehmann_prediction - self.z_noise), cmap=cm.Reds, linewidth=0, antialiased=False)

		ax.set_zlim(-0.1, 1.2)
		fig.colorbar(surf, shrink=0.5, aspect=5)
		plt.show()

	# --------------- Below here are the Score measures (MSE, R2 and Cross Validation) ------------
	def MeanSquaredError(self, target, predicted):
		mse 		= 0
		target 		= target.ravel()
		predicted 	= predicted.ravel()
		for pred, targ in zip(predicted, target):
			mse += (pred - targ)**2
		mse /= len(target) # sklearn implemented it like this somehow
		sklearn_mse = metrics.mean_squared_error(target, predicted)
		if round(sklearn_mse, 6) != round(mse, 6): # get a little bit of variance in the values
			print("THE MSE: {}\nYOUR MSE: {}".format(sklearn_mse, mse))
			return sklearn_mse
		else:
			print("The MSE: {}".format(mse))
			return mse


	def R2_Score(self, target, predicted):
		r2 			= 0
		mean_y 		= np.mean(target.ravel())
		predicted 	= predicted.ravel()
		target 		= target.ravel()
		numerator, denominator = 0, 0
		for pred, targ in zip(predicted, target):
			numerator += (pred - targ)**2
			denominator += (targ - mean_y)**2
		r2 = 1 - (numerator / denominator)
		# compare with the sklearn package
		sklearn_r2 = metrics.r2_score(target, predicted)
		if round(sklearn_r2, 6) != round(r2, 6):
			print("THE R2: {}\nYOUR R2: {}".format(sklearn_r2, r2))
			return sklearn_r2
		else:
			print("The R2: {}".format(r2))
			return r2


	def K_Fold_Cross_Validation(self, k_folds=4):
		# Reconstruct the non-mesh dataset and initialize scores
		dataset = np.zeros(shape=self.X.shape[0])
		y 	    = np.zeros(shape=self.y.shape[0])
		error   = 0
		for i in range(self.X.shape[0]): # since square matrix we only need i
			dataset[i] 	= self.X[i, i]
			y[i]		= self.y[i, i]
		#split the data into k equally sized bins
		sizes = dataset.shape[0] // k_folds

		# do the cross validation magic
		for train_runs in range(k_folds):
		# get the val indices for each run 
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

	## HOW TO MAKE PREDICTION ON TEST SET ?!

			# train the classifier on the data and evaluate on val data
			train_data, train_target = np.meshgrid(train_data, train_target) # design Matrix ecpects mesh
			X 						= CreateDesignMatrix_X(train_data, train_target, self.degree)
			franke_target 			= self.function(train_data, train_target) # y- values for beta
			beta 		= np.linalg.inv(X.T.dot(X)).dot(X.T).dot(franke_target.reshape(-1, 1))
			prediction 	= X.dot(beta).reshape(franke_target.shape[0], franke_target.shape[1])

			#target 		= self.function(val_data, val_target).ravel()
			target = franke_target.ravel()
			predicted 	= prediction.ravel()
			#print(target.shape, predicted.shape	)
			for pred, targ in zip(predicted, target):
				error += (pred - targ)**2

		error /= k_folds
		print("\nCross Validation: {}".format(error))
		return error


	def Variance(self): 
		sigma2 = self.MeanSquaredError(self.targets, self.lehmann_prediction)
		beta_var = np.linalg.inv(self.XY.T @ self.XY) * sigma2
		print("Variance of Sigma Square: {}, \nVariance of Beta: {}".format(sigma2, beta_var))


# ------------------------------ MAIN - TESTING -----------------------------------------------------
if __name__ == "__main__":
	testScikit 	= False
	testLehmann = False
	testLasso 	= False	
	testRidge 	= False	
	testVar 	= False
	testCrossValidation = False

	splitting = True

	test 		= RegressionMethods(n=120, function=FrankeFunction, degree=5, lamda=10)

	test.Lehmann_OLS(test.X_train, test.y_train, with_split=splitting, with_noise=False)
	test.Lehmann_Predictions('OLS', test.XY, with_split=splitting)
	test.MeanSquaredError(test.train_targets, test.lehmann_prediction)
	
	test.Lehmann_Ridge(test.lamda, test.X,test.y, with_split=False, with_noise=False)
	test.Lehmann_Predictions('RIDGE', test.XY, with_split=False)
	test.R2_Score(test.targets, test.lehmann_ridge)


	if testScikit:
		print("\n ################ OLS ########################")
		test.Sklearn_OLS()
		test.R2_Score(test.targets, test.sklearn_prediction)
		test.MeanSquaredError(test.targets, test.sklearn_prediction)
		print('\n Train Test OLS')
		test.Sklearn_OLS_test_train()
		test.R2_Score(test.targets, test.sklearn_pred_test_train)
	
	if testLehmann:
		print("\n ################ LEHMANN ########################")
		test.Lehmann_OLS()
		test.R2_Score(test.targets, test.lehmann_prediction)
		test.MeanSquaredError(test.targets, test.lehmann_prediction)
		print("\n Now woth noise:")
		test.Lehmann_OLS_noise(5)
		test.R2_Score(test.targets, test.lehmann_noisy_prediction)
		test.MeanSquaredError(test.targets, test.lehmann_noisy_prediction)
		print("\n TRAIN AND TEST SPLIT: ")
		test.Lehmann_OLS_test_train()
		test.R2_Score(test.train_targets, test.lehmann_pred_test_train)
		test.MeanSquaredError(test.train_targets, test.lehmann_pred_test_train)

	if testRidge:
		print("\n ################ RIDGE ########################")
		test.Sklearn_Ridge(test.lamda)
		test.R2_Score(test.targets, test.sklearn_ridge)
		test.MeanSquaredError(test.targets, test.sklearn_ridge)

		print("\n LEHMANN RIDGE:")
		for lamda in [0.01, 0.1, 1, 10, 100]:
			test.Lehmann_Ridge(lamda)
			print("Lambda Value: {}".format(lamda))
			test.R2_Score(test.targets, test.lehmann_ridge)
			test.MeanSquaredError(test.targets, test.lehmann_ridge)

	if testLasso:
		print("\n ################ LASSO ######################## \n")
		test.Sklearn_Lasso()
		test.R2_Score(test.targets, test.sklearn_lasso)
		test.MeanSquaredError(test.targets, test.sklearn_lasso)

	if testCrossValidation	:
		test.K_Fold_Cross_Validation()
	
	print(" ------------------------      -----------------------    ------------------------ \n")
	if testVar:
		test.Variance(k_folds=4)

	#test.Plot()