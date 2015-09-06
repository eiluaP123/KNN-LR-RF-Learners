import numpy as np
import sys, csv, math

class LinRegLearner(object):
	def __init__(self):
		self.Xtrain = None
		self.Ytrain = None
		
	def addEvidence(self, Xtrain, Ytrain):
		self.Xtrain = Xtrain
		self.Ytrain = Ytrain
		alpha_col = np.ones((len(self.Xtrain), 1))
		matrix = np.hstack((self.Xtrain,alpha_col))
		param = np.linalg.lstsq(matrix,self.Ytrain)
		self.coeff = param[0]
		
	def query(self, Xtest):
		alpha_col = np.ones((len(Xtest), 1))
		Xtest = np.hstack((Xtest, alpha_col))
		res = np.dot(Xtest,self.coeff)
		return res
				
if __name__=="__main__":
	reader = csv.reader(open(sys.argv[1],'rU'), delimiter=',')
	data = list(reader)
	nrows = len(data)
	ncols = len(data[1])-1
	#print nrows
	#print ncols
	Xtrain = np.zeros(((0.6*nrows),ncols))
	Ytrain = np.zeros(((0.6*nrows),1))
	Xtest = np.zeros(((0.4*nrows), ncols))
	Ytest = np.zeros((0.4*nrows))
	count=0
	index_training=0
	index_test=0
	for row in data:
		if count < (0.6*nrows):
			for i in range(ncols):
				Xtrain[index_training,i] = row[i]
			Ytrain[index_training,:]=row[i+1]
			index_training += 1
		else:
			for i in range(ncols):
				Xtest[index_test,i]=row[i]
			Ytest[index_test]=row[i+1]
			index_test += 1
		count += 1
	
	#k = int(sys.argv[2])
	learner = LinRegLearner()
	learner.addEvidence(Xtrain, Ytrain)
	Y = learner.query(Xtest)
	#print Y.shape
	print Y