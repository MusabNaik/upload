import numpy as np


def dpsearch(points,k):
	"""
	This is dynamic programming approach.
	"""
	#M = k
	points = np.sort(points,axis=0)
	L = len(points)
	M = k
	T = list(np.zeros(M+1,dtype='int'))
	T[0] = 0	#first threshold is by default always set to index 0 in trellis graph.
	T[M] = L 	#last threshold is by default always set to last number in input points.
	trellis_value = np.full((M+1,L+1),np.inf)
	trellis_backpointer = np.full((M+1,L+1),np.inf)

	# Stage 1: m=1	
	for l in range(1,L-M+2):
		trellis_value[1][l] = ((l-0)/float(L))*np.var(points[0:l])
		trellis_backpointer[1][l] = 0

	
	if(M>2):
		# Stage 2: m=2 to m=M-1
		for m in range(2,M):
			for l in range(m,L-M+m+1):
				#finding optimal path
				J_min = np.inf
				J_temp = np.inf
				for i in range(m-1,l):
					J_temp = trellis_value[m-1][i] + ((l-i)/float(L))*np.var(points[i:l])
					if J_temp < J_min:
						J_min = J_temp
						ptr = i
				
				trellis_value[m][l],trellis_backpointer[m][l] = J_min,ptr
				

	# Stage 3: m=M
	m = M
	l = L
	#finding optimal path
	J_min = np.inf
	J_temp = np.inf
	for i in range(m-1,l):
		J_temp = trellis_value[m-1][i] + ((l-i)/float(L))*np.var(points[i:l])
		if J_temp < J_min:
			J_min = J_temp
			ptr = i

	
	trellis_value[M][L] = J_min
	trellis_backpointer[M][L] = ptr
	
	
	# Backtracking
	l = L
	m = M
	while m>=2:
		T[m-1] = int(trellis_backpointer[m][l])
		l = int(trellis_backpointer[m][l])
		m = m - 1

	#Assign cluster labels
	labels = np.full(len(points),0)
	j = T[0]
	counter = 0
	for i in range(1,k+1):
		labels[j:T[i]] = counter
		j = T[i]
		counter += 1


	return labels,T


'''
def cluster_labels(points,k):
	T = dpsearch(points,k)
	labels = np.full(len(points),0)
	j = T[0]
	counter = 0
	for i in range(1,k+1):
		labels[j:T[i]] = counter
		j = T[i]
		counter += 1
	return labels,T

'''