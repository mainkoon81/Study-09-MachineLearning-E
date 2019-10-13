# Study-09-MachineLearning-E
UnsupervisedLearning

- **A. Basic Clustering**
  - K-mean, Hierarchical, DBSCAN
- **B. Model-Based Clustering**
  - Gaussian Mixture
- **C. Cluster Validation**
- **D. Dimensionality Reduction**
  - PCA, ICA

---
## 00. Min-Max_Scaler: Feature Scaling in the pre-processing data stage
 - Unbalanced features: height / weight..the unit differ you dummy! How this combination of features can describe someone ? 
 - Transform features to have a range [0,1], but what if it has outliers?? such as ridiculous max or min ??
<img src="https://user-images.githubusercontent.com/31917400/41969522-972a0526-79ff-11e8-9064-6035360353d2.jpg" />

```
def featureScaling(array):
    answer = []
    for i in array:
        value = float(i - min(array))/float(max(array)-min(array))
        answer.append(value)
    return(answer)
data = [115, 140, 175]
print featureScaling(data)
```
**`ScikitLearn` loves `numpy input`!!!!!**
```
import numpy as np 
from sklearn.preprocessing import MinMaxScaler

X = np.array([ [115.0],[140.0],[175.0] ]) # need to be float!! "[]" means.."row"

scaler = MinMaxScaler()
rescaled_X = scaler.fit_transform(X)
```
> [Note]: Which algorithms are affected by the **feature scaling** ??
 - SVM Classification =>(YES): We trade off one dimension to the other when calculating the `distances`(the **"diagonal"** decision_surf maximizing distances)
 - K-means Clustering =>(YES): Having a cluster center, and calculating the `distances` of it to all data pt..they are **"diagonal"**.
 - Linear Regression Classification =>(NO): Each feature always goes with its coefficient. What's going on with feature_A does not affect anything with the coefficient of feature_B..So they are separated.  
 - DescisionTree Classification =>(NO): No need to use diagonal decision surf. There is no trade off. 

---
## A. Basic Clustering
<img src="https://user-images.githubusercontent.com/31917400/41802151-ac0d97dc-7676-11e8-8c9f-30623f45fbbe.jpg" />
<img src="https://user-images.githubusercontent.com/31917400/41823116-0c410914-77f2-11e8-8d2f-52bbbffcec4e.jpg" />

### 1. K-mean Clustering
 - Find the groups of similar observations
   - Step_01: randomly generate the centroids (MU1, MU2,...).
   - Step_02: Allocation
     - Holding `MU_k` fixed, label each data points (which MU_k is closest?) and find the membership 'Z_ik' that **minimize SS** (create clusters around each MU_k).
   - Step_03: Updating
     - Holding `Z_ik` fixed, elect 'new MU_k' for each cluster that **minimize SS**.
   - Step_04: Iterate until they converge (no movement of points b/w clusters)
> **SS** of each data pt..(**find the membership** `Z_ik`(0/1) that minimize the SS)
 - i: each datapoint
 - k: each cluster
 <img src="https://user-images.githubusercontent.com/31917400/66716288-d20d3f80-edc3-11e9-9457-b6ca283df3ed.JPG" />

> **SS** for each cluster..(**find the center** `MU_k` that minimize the SS)
 - i: each datapoint
 - k: each cluster
 <img src="https://user-images.githubusercontent.com/31917400/49332513-0f9b1c80-f5a6-11e8-8e9f-025b18092aee.JPG" />

> Advantages:
 - it is simple, easy to implement and easy to interpret the results. 
 - it practically work well even some assumptions are broken.
> Disadvantages: 
 - **Local Minima**: It's a local hill climbing algorithm. It can give a sub-optimal solution. The output for any fixed training set can be inconsistent...Damn. The output would be very dependent on where we put our **initial cluster centers**. The more cluster centers we have, the more bad local minima we can get, so run the algorithm multiple times.
<img src="https://user-images.githubusercontent.com/31917400/41810278-e6b5e196-76f3-11e8-9fa0-1d0cb04a6975.jpg" />

 - **Hyper-spherical nature**: 
   - it only relies on distance to centroid as a definition of a cluster, thus it works poorly with clusters with different densities and cannot carve out descent clusters when their shapes are not spherical.
   - it assumes the joint distribution of features within each cluster is spherical, features within a cluster have equal variance, and also features are independent of each other.
   - it assumes balanced cluster size within the dataset, thus often produces clusters with relatively uniform size even if the input data have different cluster size. 
   - it is sensitive to outliers
```
def kmeans(dataSet, k):
	
    # Initialize centroids randomly
    numFeatures = dataSet.getNumFeatures()
    centroids = getRandomCentroids(numFeatures, k)
    
    # Initialize book keeping vars.
    iterations = 0
    oldCentroids = None
    
    # Run the main k-means algorithm
    while not shouldStop(oldCentroids, centroids, iterations):
        # Save old centroids for convergence test. Book keeping.
        oldCentroids = centroids
        iterations += 1
        
        # Assign labels to each datapoint based on centroids
        labels = getLabels(dataSet, centroids)
        
        # Assign centroids based on datapoint labels
        centroids = getCentroids(dataSet, labels, k)
        
    # We can get the labels too by calling getLabels(dataSet, centroids)
    return centroids
    
# Function: Should Stop
# -------------
# Returns True or False if k-means is done. K-means terminates either
# because it has run a maximum number of iterations OR the centroids
# stop changing.
def shouldStop(oldCentroids, centroids, iterations):
    if iterations > MAX_ITERATIONS: return True
    return oldCentroids == centroids

# Function: Get Labels
# -------------
# Returns a label for each piece of data in the dataset. 
def getLabels(dataSet, centroids):
    # For each element in the dataset, chose the closest centroid. 
    # Make that centroid the element's label.
    
# Function: Get Centroids
# -------------
# Returns k random centroids, each of dimension n.
def getCentroids(dataSet, labels, k):
    # Each centroid is the geometric mean of the points that
    # have that centroid's label. Important: If a centroid is empty (no points have
    # that centroid's label) you should randomly re-initialize it.    
```
### 2. Hierarchical & Density-Based Clustering
 - In SKLEARN, they are parts of `agglomerative clustering` component.  
<img src="https://user-images.githubusercontent.com/31917400/41822691-a7ced232-77eb-11e8-946f-40b479b843be.jpg" />

> Hierarchical Clustering Example: A Pizza company want to cluster the locations of its customers in order to determine where it should open up its new branches.

1. Hierarchical Single-link clustering: 
 - Hierarchical Clustering results in a **structure of clusters** that gives us a visual indication of how clusters relate to each other. 
 - Step01: assume each pt is already a cluster and we give each pt a label. 
 - Step02: calculate the distance b/w each pt and each other pt, then choose the smallest distances to group them into a cluster. On the side, we draw the structure tree one by one (the dendogram gives us an additional insight that might direct the results of the clustering misses) 
<img src="https://user-images.githubusercontent.com/31917400/41822846-fb1a5374-77ed-11e8-8f71-50aad55778a5.jpg" />

 - Single linkage looks at the closest point to the cluster, that can result in clusters of various shapes, thus is more prone to result in elongated shapes that are not necessarily compact or circular. 
 - Single and complete linkage follow merging heuristics that involve mainly one point. They do not pay much attention to in-cluster variance.
 - Ward's method does try to minimize the variance resulting in each merging step by merging clusters that lead to the least increase in variance in the clusters after merging. 

2. Hierarchical Complete-link clustering:....
3. Hierarchical Average-link clustering:....
4. Ward's Method:....
```
from sklearn.cluster import AgglomerativeClustering

# Ward is the default linkage algorithm...
ward = AgglomerativeClustering(n_clusters=3)
ward_pred = ward.fit_predict(df)

# using complete linkage
complete = AgglomerativeClustering(n_clusters=3, linkage="complete")
# Fit & predict
complete_pred = complete.fit_predict(df)

# using average linkage
avg = AgglomerativeClustering(n_clusters=3, linkage="average")
# Fit & predict
avg_pred = avg.fit_predict(df)
```
To determine which clustering result better matches the original labels of the samples, we can use adjusted_rand_score which is an external cluster validation index which results in a score between -1 and 1, where 1 means two clusterings are identical of how they grouped the samples in a dataset (regardless of what label is assigned to each cluster). Which algorithm results in the higher Adjusted Rand Score?
```
from sklearn.metrics import adjusted_rand_score

ward_ar_score = adjusted_rand_score(df.label, ward_pred)
complete_ar_score = adjusted_rand_score(df.label, complete_pred)
avg_ar_score = adjusted_rand_score(df.label, avg_pred)

print( "Scores: \nWard:", ward_ar_score,"\nComplete: ", complete_ar_score, "\nAverage: ", avg_ar_score)
```
Sometimes some column has smaller values than the rest of the columns, and so its variance counts for less in the clustering process (since clustering is based on distance). We normalize the dataset so that each dimension lies between 0 and 1, so they have equal weight in the clustering process. **This is done by subtracting the minimum from each column then dividing the difference(max-min) by the range.** Would clustering the dataset after this transformation lead to a better clustering?
```
from sklearn import preprocessing
normalized_X = preprocessing.normalize(df)
```
To visualize the highest scoring clustering result, we'll need to use Scipy's linkage function to perform the clusteirng again so we can obtain the linkage matrix it will later use to visualize the hierarchy.
```
# Import scipy's linkage function to conduct the clustering
from scipy.cluster.hierarchy import linkage

# Pick the one that resulted in the highest Adjusted Rand Score
linkage_type = 'ward'

linkage_matrix = linkage(normalized_X, linkage_type)

from scipy.cluster.hierarchy import dendrogram

plt.figure(figsize=(22,18))
dendrogram(linkage_matrix)

plt.show()
```
<img src="https://user-images.githubusercontent.com/31917400/41823713-c7335eca-77fc-11e8-9a72-bc83292eb7b2.jpg" />

5. Density-Based Clustering:
 - DBSCAN(Density-based Spatial Clustering of Applications with Noise) grips the pt densely packed together and labels other pt as noise. 
 - Step01: it selects a point arbitrarily, and looks at the neighbors around, and ask "Are there any other points?". If no, it's a noise. and ask "enough numbers to make a cluster?". If no, it's a noise. 
 - Step02: If we find enough number of points, we identify 'core point' and 'border point'.
 - Step03: Continue examine points..and create clusters. 
<img src="https://user-images.githubusercontent.com/31917400/41823873-f9733a66-77fe-11e8-843f-0e1375f6092c.jpg" />

```
DBSCAN(df, epsilon, min_points):
      C = 0
      for each unvisited point P in df
            mark P as visited
            sphere_points = regionQuery(P, epsilon)
            if sizeof(sphere_points) < min_points
                  ignore P
            else
                  C = next cluster
                  expandCluster(P, sphere_points, C, epsilon, min_points)

expandCluster(P, sphere_points, C, epsilon, min_points):
      add P to cluster C
      for each point P’ in sphere_points
            if P’ is not visited
                  mark P’ as visited
                  sphere_points’ = regionQuery(P’, epsilon)
                  if sizeof(sphere_points’) >= min_points
                        sphere_points = sphere_points joined with sphere_points’
                  if P’ is not yet member of any cluster
                        add P’ to cluster C

regionQuery(P, epsilon):
      return all points within the n-dimensional sphere centered at P with radius epsilon (including P)


#### Python #########################################################################################################
import numpy as numpy
import scipy as scipy
from sklearn import cluster
import matplotlib.pyplot as plt



def set2List(NumpyArray):
    list = []
    for item in NumpyArray:
        list.append(item.tolist())
    return list


def GenerateData():
    x1=numpy.random.randn(50,2)
    x2x=numpy.random.randn(80,1)+12
    x2y=numpy.random.randn(80,1)
    x2=numpy.column_stack((x2x,x2y))
    x3=numpy.random.randn(100,2)+8
    x4=numpy.random.randn(120,2)+15
    z=numpy.concatenate((x1,x2,x3,x4))
    return z


def DBSCAN(Dataset, Epsilon,MinumumPoints,DistanceMethod = 'euclidean'):
#    Dataset is a mxn matrix, m is number of item and n is the dimension of data
    m,n=Dataset.shape
    Visited=numpy.zeros(m,'int')
    Type=numpy.zeros(m)
#   -1 noise, outlier
#    0 border
#    1 core
    ClustersList=[]
    Cluster=[]
    PointClusterNumber=numpy.zeros(m)
    PointClusterNumberIndex=1
    PointNeighbors=[]
    DistanceMatrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(Dataset, DistanceMethod))
    for i in xrange(m):
        if Visited[i]==0:
            Visited[i]=1
            PointNeighbors=numpy.where(DistanceMatrix[i]<Epsilon)[0]
            if len(PointNeighbors)<MinumumPoints:
                Type[i]=-1
            else:
                for k in xrange(len(Cluster)):
                    Cluster.pop()
                Cluster.append(i)
                PointClusterNumber[i]=PointClusterNumberIndex
                
                
                PointNeighbors=set2List(PointNeighbors)    
                ExpandClsuter(Dataset[i], PointNeighbors,Cluster,MinumumPoints,Epsilon,Visited,DistanceMatrix,PointClusterNumber,PointClusterNumberIndex  )
                Cluster.append(PointNeighbors[:])
                ClustersList.append(Cluster[:])
                PointClusterNumberIndex=PointClusterNumberIndex+1
                 
                    
    return PointClusterNumber 



def ExpandClsuter(PointToExapnd, PointNeighbors,Cluster,MinumumPoints,Epsilon,Visited,DistanceMatrix,PointClusterNumber,PointClusterNumberIndex  ):
    Neighbors=[]

    for i in PointNeighbors:
        if Visited[i]==0:
            Visited[i]=1
            Neighbors=numpy.where(DistanceMatrix[i]<Epsilon)[0]
            if len(Neighbors)>=MinumumPoints:
#                Neighbors merge with PointNeighbors
                for j in Neighbors:
                    try:
                        PointNeighbors.index(j)
                    except ValueError:
                        PointNeighbors.append(j)
                    
        if PointClusterNumber[i]==0:
            Cluster.append(i)
            PointClusterNumber[i]=PointClusterNumberIndex
    return

#Generating some data with normal distribution at 
#(0,0)
#(8,8)
#(12,0)
#(15,15)
Data=GenerateData()

#Adding some noise with uniform distribution 
#X between [-3,17],
#Y between [-3,17]
noise=scipy.rand(50,2)*20 -3

Noisy_Data=numpy.concatenate((Data,noise))
size=20


fig = plt.figure()
ax1=fig.add_subplot(2,1,1) #row, column, figure number
ax2 = fig.add_subplot(212)

ax1.scatter(Data[:,0],Data[:,1], alpha =  0.5 )
ax1.scatter(noise[:,0],noise[:,1],color='red' ,alpha =  0.5)
ax2.scatter(noise[:,0],noise[:,1],color='red' ,alpha =  0.5)


Epsilon=1
MinumumPoints=20
result =DBSCAN(Data,Epsilon,MinumumPoints)

#printed numbers are cluster numbers
print result
#print "Noisy_Data"
#print Noisy_Data.shape
#print Noisy_Data

for i in xrange(len(result)):
    ax2.scatter(Noisy_Data[i][0],Noisy_Data[i][1],color='yellow' ,alpha =  0.5)
      
plt.show()

```
<img src="https://user-images.githubusercontent.com/31917400/41823968-5ac06194-7800-11e8-8b24-119cfed4d27f.jpg" />

---
## B. Model-Based Clustering(Gaussian Mixture)
### Wow, several datasets were hacked and mixed up..How to retrieve the originals? 

[Assumption]: **Each cluster follows a certain statistical distribution**.
 - In one dimension
<img src="https://user-images.githubusercontent.com/31917400/41854541-60d63b2a-7888-11e8-9389-628b0bb299e2.jpg" />

 - In two dimension
<img src="https://user-images.githubusercontent.com/31917400/41857675-fa86a348-788f-11e8-8604-81c59a812d96.jpg" />

### EM(Expectation Maximization) Algorithm for Gaussian Mixture
<img src="https://user-images.githubusercontent.com/31917400/41859683-988990f6-7894-11e8-9b42-dafa3ca365b8.jpg" />

 - Step_01. Initialization of the distributions
   - > give them the initial values(`mean`, `var`) for each of the two suspected clusters. 
     - Run 'k-means' on the dataset and choose the clusters roughly.... or randomly choose ?
     - It is indeed important that we are careful in **choosing the parameters of the initial Gaussians**. That has a significant effect on the quality of EM's result.
<img src="https://user-images.githubusercontent.com/31917400/41859693-9f6987d2-7894-11e8-8721-e133859f2636.jpg" />

 - Step_02. **Expectation I**: soft_clustering of data-pt with probabilities
   - > let's say we have 'n'points. Each pt has 2 values for each feature. Now we need to calculate the membership(probability) of each pt.
     - How to determine the membership? Just pass in your x_value, and two parameters(mean, var)...
<img src="https://user-images.githubusercontent.com/31917400/41862976-52adfc1c-789d-11e8-9dd4-6d3776c71860.jpg" />
     
 - Step_02. **Expectation II**: Estimate real **parameters** of new Gaussians, using the `weighted means & variance`
   - > the `new mean` for cluster_A, given the result of step_02(transient memberships), comes from calculating the **weighted mean** of all of the points with the same transient memberships.
     - the weighted mean does not only account for the parameters of each pt, but also account for how much it belongs.
   - > the `new var` for cluster_A, given the result of step_02(transient memberships), comes from calculating the **weighter VAR** of all of the points with the same transient memberships.       
<img src="https://user-images.githubusercontent.com/31917400/41906886-16dae142-7937-11e8-975d-e07f990de95f.jpg" />

 - Step_03. **Maximization**: Compare(overlay) the new result with the old Gaussian. We iterate these steps until it converges(no movement?).  
   - > Evaluate the `log-likelihood` which sums for all clusters.
     - the higher the value, the more sure we are that the mixer model fits out dataset.
     - the purpose is to **maximize** this value by choosing the parameters(the mixing coefficient, mean, var) of each Gaussian again and again until the value converges, reaching a maximum. 
     - What's the mixing coefficient? = mixing proportions.. they affect the **height** of the distribution.. 
<img src="https://user-images.githubusercontent.com/31917400/41907759-b96fe09a-7939-11e8-8cd6-957812adc8ce.jpg" />

```
from sklearn import mixture
gmm = mixture.GaussianMixture(n_components=3)
gmm.fit(X)
clustering = gmm.predict(X)
```
https://www.youtube.com/watch?v=lLt9H6RFO6A

http://www.ai.mit.edu/projects/vsam/Publications/stauffer_cvpr98_track.pdf

<img src="https://user-images.githubusercontent.com/31917400/41909439-ea76ac5a-793e-11e8-80f8-b17a41882b8e.jpg" />

---
## C. Cluster Validation
<img src="https://user-images.githubusercontent.com/31917400/41922451-031eb4b4-795d-11e8-95b9-3c069d974b1a.jpg" />

### 1.External Indices:__ 
<img src="https://user-images.githubusercontent.com/31917400/41922576-5570dfee-795d-11e8-8834-2fb0cda805b1.jpg" />

 - When we have the ground truth(answer-sheet or the labeled reference).
 - **ARI**(Adjusted Rand_Index) [-1 to 1]: 
   - > Note: ARI does not care what label we assign a cluster, as long as the point assignment matches that of the ground truth.
<img src="https://user-images.githubusercontent.com/31917400/41938150-29f005e8-798a-11e8-8bac-85d9c0a0c0da.jpg" />

### 2. Internal Indices:__ 
<img src="https://user-images.githubusercontent.com/31917400/41938728-c0276ab4-798b-11e8-86fd-f11e64f476df.jpg" />

 - When we don't have the ground truth.
 - **Silhouette Coefficient** [-1 to 1]:
   - There is a Silhouette Coefficient for each data-pt. We average them and get a Silhouette score for the entire clustering. We can calculate the silhouette coefficient for each point, cluster, as well as for an entire dataset. 
   - Silhouette is affected by `K`(No.of clusters) 
   - Silhouette is affected by compactness, circularity of the cluster.
   - > Note: for DBSCAN, we never use Silhouette score...(it does not care the **compact, circular clustering** because of the idea of 'noise'). Instead, we use **DBCV** for DBSCAN. http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=83C3BD5E078B1444CB26E243975507E1?doi=10.1.1.707.9034&rep=rep1&type=pdf
   - > Note: for Hierachical Clustering, it carves out the clusters well, but it's not what Silhouette can conceive of.
   
By 'K'
<img src="https://user-images.githubusercontent.com/31917400/41940187-d8c4b186-798f-11e8-9b02-1ff6fd5df88e.jpg" />

By the 'shape' of the cluster
<img src="https://user-images.githubusercontent.com/31917400/41940677-827502f2-7991-11e8-9a61-4a323e3d41e4.jpg" />

```



```
---
## D. Dimensionality Reduction
### 1. Principal Component Anaysis
<img src="https://user-images.githubusercontent.com/31917400/41972463-e1dde6f0-7a09-11e8-8a0c-735beef82dd3.jpg" />

We can't possibly come up with coordinate system shifted, rotated from the original to obtain the **one dimensionality**. PCA specializes on **'shifts'** and **'rotation'** for the coordinate system.

If our given data is of any shape whatsoever, PCA finds a **new coordinate system** obtained from the original by translation or rotation.  
 - It moves the **center** of the coordinate system with the center of the dataset.
 - It moves the X-axis into the principal axis of the variation where we see the **most variation** relative to all the data-pt.
 - It moves the y-axis down the road into the orthogonal(less important directions of variation). 

> **What defines the two principal directions(the two orthogonal vectors)?** to kill dimensionality, multicollinearity...
> - 1.Find the center of the dataset (mean??)
> - 2.Find the two principal axis of variation (Eigenvectors)
>   - The measure of the orthogonality: Do the 'dot-product' of these two vectors, we should get 'zero'.  
> - 3.Find the spread values (giving importance to our vectors) for the two axis? (Eigenvalues)
<img src="https://user-images.githubusercontent.com/31917400/42023655-4e221f58-7ab8-11e8-9332-229d9417d1d3.jpg" />

## Compression while preserving all information!!!! Get rid of multicollinearity!!!!
Let's say we have a large number of measurable features, but we know there are a small number of underlying **latent features** that contain most of information. What's the best way to condense those features? 
# The new variable is a linear-Combinations using those features! But the game changer is the Cov-matrix!!
<img src="https://user-images.githubusercontent.com/31917400/42027356-6bdeba52-7ac1-11e8-9c40-ff5175546dbf.jpg" />

 - How to find the principal component or the direction that capturing the maximal variance (the corresponding Eigenvector of Cov_matrix) ? 
   - the amount of **information loss** is equal to the distances b/w the component line(the new tranformed values) and a given pt, so find the component line that minimizes the information loss. This component line is the Eigenvector of pxp **Cov-Matrix ?**.  
<img src="https://user-images.githubusercontent.com/31917400/42027169-fb884a48-7ac0-11e8-935f-1ded1892b17e.jpg" />

 - How to give an insight on **which features** drive the most impact(capturing the major pattern - the largest Eigenvalue of Cov-matrix) ?
<img src="https://user-images.githubusercontent.com/31917400/42032945-1be1571e-7ad3-11e8-942d-3a7767e40b20.jpg" />
<img src="https://user-images.githubusercontent.com/31917400/42599845-724493b6-8558-11e8-9455-e76d4a91715a.jpg" />

<img src="https://user-images.githubusercontent.com/31917400/42033668-7f2ca6b4-7ad5-11e8-8eea-b410e21e512b.jpg" />
<img src="https://user-images.githubusercontent.com/31917400/42037658-e1726178-7ae0-11e8-9f85-b00d34e1cf40.jpg" />

 - [Usage]
   - When we want to examine **latent features** driving the patterns in our complex data
   - Dimensionality Reduction
     - Visualizing high-dimensional data(projecting the two features down to the first PC-line and leave them as scatters, then use K-means) 
     - Reducing **noises** by discarding unimportant PC. 
     - Pre-processing before using any other algorithms by reducing the demensionality of inputs.   

Ex> facial recognition why?
 - **Mega pixels:** pictures of human faces in general have high input dimensionality
 - **Eyes, nose, mouth:** Human faces have general patterns that could be captured in smaller number of dimensions.
 - In this example, the original dimensionality of the pic is: "1288 rows x 1850 features" plus "7 classes".
```
from time import time
import logging
import pylab as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# Download the data, if not already on disk and load it as numpy arrays
lfw_people = fetch_lfw_people('data', min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape
np.random.seed(42)

# for machine learning we use the data directly (as relative pixel
# position info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print( "n_classes: %d" % n_classes)
```
<img src="https://user-images.githubusercontent.com/31917400/42043299-04024a16-7aee-11e8-80b1-5a9b70efb817.jpg" />

### Eigenvalue and Eigenvector
A matrix is a linear transformation tool and focus on **mapping of a vector**. It can transform the **magnitude** and the **direction** of a vector into **lower dimension**! `tranformation matrix * Eigen vector = Scaled vector!!`
<img src="https://user-images.githubusercontent.com/31917400/43229747-c51a68ee-905d-11e8-89cb-1ac7657e61ed.jpg" />

### 2. RandomProjection
 - Computationally more efficient than PCA.
   - handle even more features than PCA (with a decrease in quality of projection, however.)
 - Premise
   - Simply reduce the size of dimensions in our dataset by **multiplying it by a random matrix**.
 - Where does the **'k'**(reduced dimensions) come from?
   - This algorithm extra cares about the distances b/w points. 
   - We have a certain level of guarantee that the distances will be a bit distorted, but can be preserved! 
     - the distance b/w two pt in the projection squared would be squeezed by.....  
   - The algorithm work either by setting a number of components we want(**'k'**) or by specifying a value for 'epsilon' and calculate a conservative value for **'k'**, and gives a new dataset.
<img src="https://user-images.githubusercontent.com/31917400/42087854-65fb21ce-7b8f-11e8-8acd-9995711cb404.jpg" />
<img src="https://user-images.githubusercontent.com/31917400/42087862-6873453a-7b8f-11e8-82f5-f04b4c853db7.jpg" />

```
from sklearn import random_projection
rp = random_projection.SparseRandomProjection(n_components='auto', eps=0.1) 

new_X = rp.fit_transform(X)
```
### 3. Independent Component Analysis
While PCA works to maximize 'var', ICA tries to isolate the independent sources that are mixed in the dataset.
 - EX> blind source separation: Restoring the original signals..
<img src="https://user-images.githubusercontent.com/31917400/42096573-991bf916-7bad-11e8-96f3-cf86496aacf1.jpg" />

 - To produce the original signal `S`, ICA estimate the best `W` that we can multiply by `X`.. 
 - ICA assumes
   - the features are mixtures of independent sources
   - the components must have **non-Gaussian** distributions.
   - the Central_Limit_Theorem says the distribution of a sum of independent variables(or sample means) tends towards the Gaussian.  
<img src="https://user-images.githubusercontent.com/31917400/42100267-a86bbdfc-7bb7-11e8-8d44-c8573b73e74e.jpg" />

```
from sklearn.decomposition import FastICA
X = list(zip(signal_1, signal_2, signal_3))
ica = FastICA(n_components=3)

components = ica.fit_transform(X)  ## here, these objects contain the independent components retrieved via ICA 
```

[Note]
 - 1.Let’s mix two random sources A and B. At each time, in the following plot(1), the value of A is the abscissa(x-axis) of the data point and the value of B is their ordinates(Y-axis).
 - 2.Let take two linear mixtures of A and B and plot(2) these two new variables.
 - 3.Then if we whiten the two linear mixtures, we get the plot(3)
   - the variance on both axis is now equal 
   - the correlation of the projection of the data on both axis is 0 (meaning that the covariance matrix is diagonal and that all the diagonal elements are equal). 
   - Then applying ICA only mean to “rotate” this representation back to the original A and B axis space.
   - The **whitening process** is simply a `linear change of coordinate` of the mixed data. Once the ICA solution is found in this “whitened” coordinate frame, we can easily reproject the ICA solution back into the original coordinate frame. 
   - **whitening** is basically a de-correlation transform that converts the covariance-matrix into an identity matrix
 <img src="https://user-images.githubusercontent.com/31917400/57101521-5473dc80-6d19-11e9-9a6b-eca58accaef7.jpg" />

We can imagine that ICA rotates the **whitened matrix** back to the original (A,B) space (first scatter plot above). It performs the rotation by **minimizing the Gaussianity of the data** projected on both axes (fixed point ICA). For instance, in the example above, the projection on both axis is quite Gaussian (i.e., it looks like a bell shape curve). By contrast, the projection in the original A, B space far from gaussian. 
 - By rotating the axis and minimizing Gaussianity of the projection in the first scatter plot, ICA is able to recover the original sources which are statistically independent (this property comes from the central limit theorem which states that any linear mixture of 2 independent random variables is more Gaussian than the original variables). 
   - the function kurtosis gives an indication of the gaussianity of a distribution (but the fixed-point ICA algorithm uses a slightly different measure called negentropy). 
<img src="https://user-images.githubusercontent.com/31917400/57102033-6a35d180-6d1a-11e9-9ec5-ad80486ff90b.jpg" />

We dealt with only 2 dimensions. However ICA can deal with an arbitrary high number of dimensions. Let’s consider 128 EEG electrodes for instance. The signal recorded in all electrode at each time point then constitutes a data point in a 128 dimension space. After whitening the data, ICA will “rotate the 128 axis” in order to minimize the Gaussianity of the projection on all axis (note that unlike PCA the axis do not have to remain orthogonal). What we call ICA components is the matrix that allows projecting the data in the initial space to one of the axis found by ICA. The weight matrix is the full transformation from the original space. 































