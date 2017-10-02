%matplotlib inline
from numpy import random, array

#Create fake income/age clusters for N people in k clusters
numClusters = 0
def createClusteredData(N, k):
    random.seed(10)
    numClusters = k
    pointsPerCluster = float(N)/k #5 clusters with 20 data points per cluster
    X = []
    for i in range (k): #loop 5 times
        incomeCentroid = random.uniform(20000.0, 200000.0) #Samples are uniformly distributed over the half-open interval [low, high) (includes low, but excludes high). In other words, any value within the given interval is equally likely to be drawn by uniform
        ageCentroid = random.uniform(20.0, 70.0)
        for j in range(int(pointsPerCluster)):
            X.append([random.normal(incomeCentroid, 10000.0), random.normal(ageCentroid, 2.0)])
    X = array(X)
    return X

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from numpy import random, float

maxClusters = 10
ptsPerCluster = 100
for i in range(1, maxClusters+1):
    data = createClusteredData(ptsPerCluster, i)
    model = KMeans(n_clusters=i)
    model = model.fit(scale(data))  #The preprocessing.scale() algorithm puts your data on one scale. This is helpful with largely sparse datasets. In simple words, your data is vastly spread out. For example the values of X maybe like so:

    plt.figure(figsize=(8, 6))
    plt.scatter(data[:,0], data[:,1], c=model.labels_.astype(float))
    plt.title("# of clusters: %s" % (i, ))
    plt.xlabel('income')
    plt.ylabel('age')
    plt.show()






