import numpy as np
import random, csv, sys

class RandomForestLearner(object):
	def __init__(self, k):
		self.k = k
		self.tree = None
		self.forest = []
		self.index = 0
		
	def query(self, test):
		Y = np.zeros(len(test))
		for i in range (len(test)):
			Y[i] = self.queryTree(test[i])
		return Y
			
	def queryTree(self, test):
		Y = 0.0
		for k in range(self.k):
			tree = self.forest[k]
			node = 0
			while tree[node,1] != -1:
				if test[tree[node,1]] <= tree[node,2]:
					node = tree[node,3]
				else:
					node = tree[node,4]
			Y += tree[node,2]
		if self.k==1:
			return Y
		else:
			return Y/k
		
	def createTree(self, data):
		#print "random data"
		#print data
		tree_index = self.index
		nrows = len(data)
		ncols = len(data[0])
		nfeatures = ncols-1
		if len(data) == 1:
			y_val = data[0,ncols-1]
			self.tree[self.index,:] = [self.index,-1, y_val, -1, -1]
		else:
			#left_tree = np.zeros((0,len(data[0])))
			#right_tree = np.zeros((0,len(data[0])))
			left_tree= np.zeros((len(data),len(data[0])))
			right_tree= np.zeros((len(data),len(data[0])))
			feature_sel = random.randrange(0,ncols-1)
			#print "feature_sel", feature_sel
			#randint giving max depth recursion error
				
			mark = False
			value_f = data[0,feature_sel]
			for i in range(len(data)):
				if data[i,feature_sel] == value_f:
					mark = True
				else:
					mark = False
					break
			#print i
			#print mark
			if mark == True:
				mean = np.mean(data[:,-1])
				self.tree[self.index,:] = [self.index,-1,mean,-1,-1]
			else:
				value1 = data[random.randrange(0,len(data)),feature_sel]
				value2 = data[random.randrange(0,len(data)),feature_sel]
				split_val = (value1 + value2)/2
				#print "split_val", split_val
				self.tree[self.index,:] = [self.index, feature_sel, split_val, self.index +1, -1]
				#print self.tree
				lt = 0
				rt = 0
				for i in range(0,len(data)):
					if data[i,feature_sel] <= split_val:
						left_tree[lt,:] = data[i,:]
						#print "left_tree"
						#print left_tree
						lt += 1
					else:
						right_tree[rt,:] = data[i,:]
						#print "right_tree"
						#print right_tree
						rt += 1
						
				self.index += 1				
				self.createTree(left_tree[0:lt])
				if(rt>=1):
					self.index += 1
					ptr_rtree_root = self.index
					self.createTree(right_tree[0:rt])
					self.tree[tree_index,-1] = ptr_rtree_root		
				#print self.tree
			
	def addEvidence(self, Xtrain, Ytrain):
		data = np.hstack([Xtrain,Ytrain])
		#print "data"
		#print data
		for k in range(self.k):
			self.index = 0
			random.shuffle(data)
			traindata = data[:len(data)*0.6]
			self.tree = np.zeros((traindata.size,5))
			self.createTree(traindata)
			self.forest.append(self.tree[0:self.index+1])
			self.tree = None
			traindata = None		
	
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
	coeff = 0.0
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
		
	k = int(sys.argv[2])
	learner = RandomForestLearner(k)
	learner.addEvidence(Xtrain,Ytrain)
	Y = learner.query(Xtest)
	print "print Y"
	print Y