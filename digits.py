from sklearn.cluster import KMeans
import thresholding
import numpy as np
from statistics import mode 
import timeit
import sklearn
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph


def load_digits(path):
	csvdata = np.array(pd.read_csv(path),dtype=np.int16)
	labels = csvdata[:,0]
	data = csvdata[:,1:785]
	#data = csvdata[:,0:2]
	#labels = csvdata[:,-1]
	return data,labels


def display_digit(row_values,label):
	plt.figure()
	fig = plt.imshow(row_values.reshape(28,28))
	fig.set_cmap('gray_r')
	fig.axes.get_xaxis().set_visible(False)
	fig.axes.get_yaxis().set_visible(False)
	plt.title("Digit:"+str(label))
	plt.show()



#This function calculates and returns Adacency, Degree and Laplacian matrices.
def Laplacian_matrix(data,knn,gamma):
	print(data.shape)
	A = np.full((data.shape[0],data.shape[0]),0.0,dtype=np.float64)
	#L = np.full((data.shape[0],data.shape[0]),0.0,dtype=np.float64)
	if knn!=-1:
		A = kneighbors_graph(data, n_neighbors=knn, mode='distance').toarray()
		D = np.diagflat((np.full((1,data.shape[0]),knn,dtype=np.float64)))
		print("Weight Matrix created using KNN : ",knn)
	elif gamma!=-1:
		A = sklearn.metrics.pairwise.rbf_kernel(data, gamma=gamma)		#RBF Kernel for constructing similarity matrix
		#D = np.diagflat(np.full((1,data.shape[0]),(data.shape[0])-1,dtype=np.float64))
		D = np.diagflat(np.count_nonzero(A,axis=1))
		print("Weight Matrix created using gamma : ",gamma)
	print("Dimensions of Similarity Matrix:	",A.shape)
	print(A)
	print("Dimensions of Degree Matrix:	",D.shape)
	print(D)
	L = np.subtract(D,A)
	print("Dimensions of Laplacian Matrix:	",L.shape)
	print(L)
	print("Laplacian Matrix created . . .")
	return A, D, L



#This function calculates and returns eigenvectors and eigenvalues of Laplacian Matrix.
def calculate_eigenpairs(L):
	print('Calculating eigenvector and eigenvalues . . .')
	vals, vecs = (np.linalg.eig(L))
	vals = vals.real
	vecs = vecs.real
	print('Eigenvalues values sorted in ascending order . . .')
	vecs = vecs[:,np.argsort(vals)]
	vals = vals[np.argsort(vals)]
	return vecs,vals



def cluster(clusters,vecs,cutstart,cutend,labels):
	kmeans = KMeans(n_clusters=clusters,init='random',algorithm='full')
	
	data = vecs[:,cutstart:cutend]
	length = len(data)
	
	start = timeit.default_timer()
	kmeans.fit(data)	
	end = timeit.default_timer()
	time_kmeans = ((end-start))

	
	start = timeit.default_timer()
	thresholding_labels,T = thresholding.dpsearch(data,clusters)
	end = timeit.default_timer()
	time_thresholding = ((end-start))

	kmeans_result = kmeans.labels_
	list_data = data.reshape(-1,).tolist()
	data_copy1 = data.reshape(-1,).tolist()
	data_copy2 = data.reshape(-1,).tolist()
	sorted_data = np.sort(data,axis=0)
	list_sorted_data = sorted_data.reshape(-1,).tolist()
	
	#Assign data to clusters		
	list_kmeans_clusters = []
	temp = []
	for k in range (0,clusters):
		temp = []
		for i in range(0,length):
			if kmeans_result[i]==k:
				temp.append(list_data[i])
		list_kmeans_clusters.append(temp)


	
	
	list_thresholding_clusters = []
	for i in T:
		if i==0:
			start = 0
			continue
		else:
			list_thresholding_clusters.append(list_sorted_data[start:i])
			start = i

	
	
	list_kmeans_clusters_groundlabels = []
	temp = []
	for c in list_kmeans_clusters:
		temp = []
		for item in c:
			for i in range(0,length):
				if(item==data_copy1[i]):
					temp.append(labels[i])
					data_copy1[i] = np.inf
					break
		list_kmeans_clusters_groundlabels.append(temp)


	list_thresholding_clusters_groundlabels = []
	temp = []
	for c in list_thresholding_clusters:
		temp = []
		for item in c:
			for i in range(0,length):
				if(item==data_copy2[i]):
					temp.append(labels[i])
					data_copy2[i] = np.inf
					break
		list_thresholding_clusters_groundlabels.append(temp)


	
	inputlabel_frequency = []
	for i in range(0,10):
		inputlabel_frequency.append(0)

	for i in labels:
		inputlabel_frequency[i] += 1


	KMeans_cluster_mostfrequentlabel = []
	for c in list_kmeans_clusters_groundlabels:
		try:
			KMeans_cluster_mostfrequentlabel.append(mode(c))
		except(Exception):
			KMeans_cluster_mostfrequentlabel.append(-1)


	KMeans_cluster_mostfrequentlabel_frequency = []
	counter = 0
	i = 0
	for c in list_kmeans_clusters_groundlabels:
		counter = 0
		for item in c:
			if item==KMeans_cluster_mostfrequentlabel[i]:
				counter += 1
		i += 1
		KMeans_cluster_mostfrequentlabel_frequency.append(counter)


	KMeans_cluster_purity = []
	for i in range(0,clusters):
		KMeans_cluster_purity.append(KMeans_cluster_mostfrequentlabel_frequency[i]/inputlabel_frequency[KMeans_cluster_mostfrequentlabel[i]])


	KMeans_total_cluster_purity = sum(KMeans_cluster_purity)
	KMeans_percentage_purity = (KMeans_total_cluster_purity/clusters)*100.0



	thresholding_cluster_mostfrequentlabel = []
	for c in list_thresholding_clusters_groundlabels:
		try:
			thresholding_cluster_mostfrequentlabel.append(mode(c))
		except(Exception):
			thresholding_cluster_mostfrequentlabel.append(-1)


	thresholding_cluster_mostfrequentlabel_frequency = []
	counter = 0
	i = 0
	for c in list_thresholding_clusters_groundlabels:
		counter = 0
		for item in c:
			if item==thresholding_cluster_mostfrequentlabel[i]:
				counter += 1
		i += 1
		thresholding_cluster_mostfrequentlabel_frequency.append(counter)




	thresholding_cluster_purity = []
	for i in range(0,clusters):
		thresholding_cluster_purity.append(thresholding_cluster_mostfrequentlabel_frequency[i]/inputlabel_frequency[thresholding_cluster_mostfrequentlabel[i]])


	thresholding_total_cluster_purity = sum(thresholding_cluster_purity)
	thresholding_percentage_purity = (thresholding_total_cluster_purity/clusters)*100.0


	#print("\nOriginal Data:	",list_data)
	#print("Sorted Data:	",list_sorted_data)

	#print("Clustering using KMeans . . .")
	#print("KMeans-Labels:	",kmeans.labels_)
	#print("K-Means Clusters:	",list_kmeans_clusters)
	#print("K-means Digit Labels:	",list_thresholding_clusters_groundlabels)
	#print("InputFreq:	",inputlabel_frequency)
	#print("Most frequent Cluster Label:	",thresholding_cluster_mostfrequentlabel)
	#print("Frequency of most frequent label:	",thresholding_cluster_mostfrequentlabel_frequency)
	#print("Purity:	",thresholding_cluster_purity)
	#print("Percentage Total:	",thresholding_percentage_purity)
	#print("Clustering using thresholding . . .")
	#print("Thresholding-Labels:	",thresholding_labels)
	#print("Thresholding Clusters:	",list_thresholding_clusters)
	#print("Thresholding Digit Labels:	",list_thresholding_clusters_groundlabels)
	
	
	thresholding_metrics = [time_thresholding,T,thresholding_labels,list_thresholding_clusters,thresholding_percentage_purity]
	kmeans_metrics = [time_kmeans,kmeans.labels_,list_kmeans_clusters,KMeans_percentage_purity]

	
	return thresholding_metrics,kmeans_metrics


def optimality_ratio(list_thresholding_clusters,list_kmeans_clusters):
	print("\nCalculating cluster withinss and optimality ratio . . .")
	thresholding_withinss = 0.0
	kmeans_withinss = 0.0
	for i in list_thresholding_clusters:
		thresholding_withinss += ((np.var(i))*len(i)) 
		
	for i in list_kmeans_clusters:
		kmeans_withinss += ((np.var(i))*len(i))

	if(thresholding_withinss==0):
		ratio = 'thresholding_withinss=0'
		return thresholding_withinss,kmeans_withinss,ratio
	ratio = (kmeans_withinss - thresholding_withinss)/thresholding_withinss
	#ratio = (100.0*(kmeans_withinss-thresholding_withinss))/thresholding_withinss
	return thresholding_withinss,kmeans_withinss,ratio



k = 10
flag = 1
knn=-1
gamma=-1.0
#method = input("Select KNN or RBF : ")
#if method.upper()=='KNN':
#	knn = int(input("Enter knn value = "))
#elif method.upper()=='RBF':
gamma = float(input("Enter value of gamma for RBF kernel = "))
path=os.getcwd()+"/digits.csv"
print("Loading Dataset. . .")
data,labels = load_digits(path)																																										
num_start = int(input("Enter starting number of digits to cluster = "))
num_end = int(input("Enter last number of digits to cluster = "))
data = data[num_start:num_end]
labels = labels[num_start:num_end]	
print("Dimensions of dataset:	",data.shape)
A,D,L = Laplacian_matrix(data,knn,gamma)			#ReturnsAdacency, Degree and Laplacian matrices.
vecs,vals = calculate_eigenpairs(L)	
metrics = []
try:
	for i in range(2,k+1):
		clusters=i
		print("\nClusters : ",clusters)
		#path = "/home/nachiket/M.Sc./Pattern_Recognition/Assignment-2/Dataset/halfkernel.csv"
		choice ='yes'
		while choice=='yes' and flag==1:
			print("Total number of eigen values = ",vals.shape[0])
			index = int(input("Enter number of eigenvalues to display(ascending order) = "))
			print(vals[0:index])
			#cutstart = int(input())
			#cutend = int(input())
			fiedler = int(input("\nSelect fiedler vector(Index starting from 0....) = "))
			cutstart=fiedler
			cutend=cutstart+1
			choice = input("Select different fiedler vector? . . .(yes/no) = ")
		flag = 0
		thresholding_metrics,kmeans_metrics = cluster(clusters,vecs,cutstart,cutend,labels)#Returns class membership values after performing k-means on data in k-dimension.
		thresholding_withinss,kmeans_withinss,ratio = optimality_ratio(thresholding_metrics[3],kmeans_metrics[2])
		result = np.array([clusters,thresholding_metrics[0],kmeans_metrics[0],thresholding_withinss,kmeans_withinss,ratio,kmeans_metrics[3],thresholding_metrics[4]])
		metrics.append(result)
		
		print("\nThresholding Time:	",thresholding_metrics[0])
		print("Kmeans Time:		",kmeans_metrics[0])
		print("thresholding_withinss:	",thresholding_withinss)
		print("kmeans_withinss:		",kmeans_withinss)
		print("Optimality Ratio:	",ratio)
		print("KMeans Percentage Purity:	",kmeans_metrics[3])
		print("Thresholding Percentage Purity:	",thresholding_metrics[4])
		print("\n")
		#plot2D(data,result[3],inputlabel,"twogaussians42","Thresholding")
		#plot2D(data,result[4],inputlabel,"twogaussians42","Kmeans")

except (KeyboardInterrupt, SystemExit, Exception):
 	raise
finally:
	path = os.getcwd()+"/out.csv"
	df = pd.DataFrame(metrics)
	df.columns = ['Clusters','Thresholding Time(sec)','Kmeans Time(sec)','Thresholding Withinss','Kmeans Withinss','Optimality Ratio','KMeans Purity(%)','Thresholding Purity(%)']
	df.to_csv(path,index=False,columns=['Clusters','Thresholding Time(sec)','Kmeans Time(sec)','Thresholding Withinss','Kmeans Withinss','Optimality Ratio','KMeans Purity(%)','Thresholding Purity(%)'])

