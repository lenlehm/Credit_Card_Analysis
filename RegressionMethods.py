import random
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model, metrics
from sklearn.preprocessing import PolynomialFeatures
#from sklearn.model_selection import train_test_split

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
	def __init__(self, function, n=100, lamda=10, mu=0, sigma=0.4, degree=2):
		# setup noise level with mean, variance and number of samples to draw
		self.n 					= n
		self.noise 				= np.random.normal(mu, sigma, n)
		self.degree 			= degree
		self.function 			= function # here we will only deal with the Franke Function
		# add the regularization parameter for Lasso and Ridge
		self.lamda 				= lamda

		# generate data
		X 						= np.random.rand(n)
		y 						= np.random.rand(n)
		self.X, self.y 			= np.meshgrid(X, y)
		self.z 					= self.function(self.X, self.y) + self.noise
		self.targets 			= self.function(self.X, self.y)

		# placeholder for all the predictions throughout the models
		self.sklearn_prediction		  = None
		self.lehmann_prediction 	  = None
		self.sklearn_ridge 			  = None
		self.sklearn_lasso 			  = None
		self.lehmann_noisy_prediction = None
		self.XY 					  = None # needed for variance calculation

	def Sklearn_OLS(self):
		polynom 				= PolynomialFeatures(degree=self.degree)
		XY 						= polynom.fit_transform(np.array([self.X.ravel(), self.y.ravel()]).T)
		regression 				= linear_model.LinearRegression(fit_intercept=False)
		regression.fit(XY, self.z.reshape(-1, 1))
		self.sklearn_prediction = regression.predict(XY)
		self.sklearn_prediction = self.sklearn_prediction.reshape(self.z.shape[0], self.z.shape[1])

	def Lehmann_OLS(self):
		# creating the OLS according to the lecture slides
		X 		= CreateDesignMatrix_X(self.X, self.y, self.degree)
		beta 	= np.linalg.inv(X.T.dot(X)).dot(X.T).dot(self.targets.reshape(-1, 1))
		self.XY = X
		prediction = X.dot(beta)
		self.lehmann_prediction = prediction.reshape(self.n, self.n)

	def Lehmann_OLS_noise(self, noise_factor):
		self.z 		= self.function(self.X, self.y) + (noise_factor * self.noise)
		# creating the OLS according to the lecture slides
		X 		= CreateDesignMatrix_X(self.X, self.y, self.degree)
		self.XY = X
		beta 	= np.linalg.inv(X.T.dot(X)).dot(X.T).dot(self.z.reshape(-1, 1))
		prediction = X.dot(beta)
		self.lehmann_noisy_prediction = prediction.reshape(self.n, self.n)

	def Sklearn_Ridge(self): 
		if self.lamda == None :
			raise ValueError("No lambda value set for Ridge regression.")
		polynom 				= PolynomialFeatures(degree=self.degree)
		XY 						= polynom.fit_transform(np.array([self.X.ravel(), self.y.ravel()]).T)
		regression 				= linear_model.Ridge(fit_intercept=True, alpha=self.lamda)
		regression.fit(XY, self.z.reshape(-1, 1))
		#beta = regression.coef_
		#beta[0] = regression.intercept_
		self.sklearn_ridge = regression.predict(XY)
		self.sklearn_ridge = self.sklearn_ridge.reshape(self.z.shape[0], self.z.shape[1])

	def Sklearn_Lasso(self):
		if self.lamda == None:
			raise ValueError("No lambda value set for Lasso regression.")

		polynom 				= PolynomialFeatures(degree=self.degree)
		XY 						= polynom.fit_transform(np.array([self.X.ravel(), self.y.ravel()]).T)
		regression 				= linear_model.Lasso(fit_intercept=True, max_iter=10000, alpha=self.lamda)
		regression.fit(XY, self.z.reshape(-1, 1))
		#beta = regression.coef_
		#beta[0] = regression.intercept_
		self.sklearn_lasso = regression.predict(XY)
		self.sklearn_lasso = self.sklearn_lasso.reshape(self.z.shape[0], self.z.shape[1])

	def Plot(self): 
		fig 	= plt.figure(figsize=(12,9))
		ax 		= fig.gca(projection='3d')
		surf 	= ax.plot_surface(self.X, self.y, self.z, cmap=cm.Greens, linewidth=0, antialiased=False)
		if self.sklearn_prediction.any() != None: 
			surf = ax.plot_surface(self.X + 1, self.y, self.sklearn_prediction, cmap=cm.Oranges, linewidth=0, antialiased=False)
		if self.lehmann_prediction.any() != None: 
			surf = ax.plot_surface(self.X, self.y + 1, self.lehmann_prediction, cmap=cm.Blues, linewidth=0, antialiased=False)
			surf = ax.plot_surface(self.X + 1, self.y + 1, abs(self.lehmann_prediction - self.z), cmap=cm.Reds, linewidth=0, antialiased=False)

		ax.set_zlim(-0.1, 1.2)

		fig.colorbar(surf, shrink=0.5, aspect=5)
		plt.show()

	def MeanSquaredError(self, target, predicted):
		mse 		= 0
		target 		= target.ravel()
		predicted 	= predicted.ravel()
		for pred, targ in zip(predicted, target):
			mse += (pred - targ)**2
		mse /= 10000 # sklearn implemented it like this somehow
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

	def Variance(self): 
		sigma2 = self.MeanSquaredError(self.targets, self.lehmann_prediction)
		beta_var = np.linalg.inv(self.XY.T @ self.XY) * sigma2
		print("Variance of Sigma Square: {}, \nVariance of Beta: {}".format(sigma2, beta_var))

# ------------------------------ MAIN - TESTING -----------------------------------------------------
if __name__ == "__main__":
	testScikit 	= False
	testLehmann = True
	testLasso 	= False	
	testRidge 	= False	
	testVar 	= False

	test 		= RegressionMethods(n=100, function=FrankeFunction, degree=5)
	
	if testScikit:
		print("\n ################ OLS ########################")
		test.Sklearn_OLS()
		test.R2_Score(test.targets, test.sklearn_prediction)
		test.MeanSquaredError(test.targets, test.sklearn_prediction)
	
	if testLehmann:
		print("\n ################ LEHMANN ########################")
		test.Lehmann_OLS()
		test.R2_Score(test.targets, test.lehmann_prediction)
		test.MeanSquaredError(test.targets, test.lehmann_prediction)
		print("\n")
		test.Lehmann_OLS_noise(5)
		test.R2_Score(test.targets, test.lehmann_noisy_prediction)
		test.MeanSquaredError(test.targets, test.lehmann_noisy_prediction)

	if testRidge:
		print("\n ################ RIDGE ########################")
		test.Sklearn_Ridge()
		test.R2_Score(test.targets, test.sklearn_ridge)
		test.MeanSquaredError(test.targets, test.sklearn_prediction)

	if testLasso:
		print("\n ################ LASSO ######################## \n")
		test.Sklearn_Lasso()
		test.R2_Score(test.targets, test.sklearn_lasso)
		test.MeanSquaredError(test.targets, test.sklearn_lasso)
	
	print(" ------------------------      -----------------------    ------------------------ \n")
	if testVar:
		test.Variance()
	
	#test.Plot()