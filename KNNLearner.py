import numpy as np
import sys, csv, math

class KNNLearner(object):
	def __init__(self,k):
		self.k = k
		
	def addEvidence(self, Xtrain, Ytrain):
		self.Xtrain = Xtrain
		self.Ytrain = Ytrain
		
	def query(self,Xtest):
		Y = np.zeros(len(Xtest))
		#dis = 0.0
		for x in range(len(Xtest)):
			distance = np.zeros((len(self.Xtrain),1))
			
			for i in range(len(self.Xtrain)):
				#distance[i] = math.sqrt(dis)
				distance[i,0] = self.calcDistance(Xtest[x],self.Xtrain[i])
				#distance[i,0] += ((Xtest[i] - self.Xtrain[i]) ** 2)
				#distance[i,0] = math.sqrt(distance[i,0])
			result = self.Ytrain[distance[:,0].argsort()]
			sel_neighbors = np.zeros(self.k)
			for n in range(self.k):
				sel_neighbors[n] = result[n]
			Y[x] = np.mean(sel_neighbors)
		#print Y.shape
		return Y
	
	def calcDistance(self, testrow=[], trainrow=[]):
		dis = 0.0
		for i in range(len(testrow)):
			dis += ((testrow[i] - trainrow[i]) ** 2)
			
		return math.sqrt(dis)
		
if __name__=="__main__":
	reader = csv.reader(open(sys.argv[1],'rU'), delimiter=',')
	data = list(reader)
	#print data
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
		
	#print Xtrain
	#print Ytrain
	
	k = int(sys.argv[2])
	learner = KNNLearner(k)
	learner.addEvidence(Xtrain,Ytrain)
	Y = learner.query(Xtest)
	print Y