# Study-09-MachineLearning-E
UnsupervisedLearning

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
> **SS** of each data pt..(find the membership `Z_ik` that minimize the SS)

![01](http://www.sciweavers.org/upload/Tex2Img_1529748773/render.png) `Z_ik = 0/1`

> **SS** for each cluster..(find the `MU_k` that minimize the SS)

![02](http://www.sciweavers.org/upload/Tex2Img_1529755147/render.png)

> Limitation: 
 - **Local Minima**: It's a local hill climbing algorithm. It can give a sub-optimal solution. The output for any fixed training set can be inconsistent...Damn. The output would be very dependent on where we put our **initial cluster centers**. The more cluster centers we have, the more bad local minima we can get, so run the algorithm multiple times.
<img src="https://user-images.githubusercontent.com/31917400/41810278-e6b5e196-76f3-11e8-9fa0-1d0cb04a6975.jpg" />

 - **Hyper-spherical nature**: it only relies on distance to centroid as a definition of a cluster, thus it cannot carve out descent clusters when their shapes are not spherical.
```



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

 - Step_02. **soft_clustering** of data-pt
   - > let's say we have 'n'points. Each pt has 2 values for each feature. Now we need to calculate the membership(probability) of each pt.
     - How to determine the membership? Just pass in your x_value, and two parameters(mean, var)...
<img src="https://user-images.githubusercontent.com/31917400/41862976-52adfc1c-789d-11e8-9dd4-6d3776c71860.jpg" />
     
 - Step_03. Estimate real **parameters** of new Gaussians, using the result of the soft_clustering
   - > the `new mean` for cluster_A, given the result of step_02(transient memberships), comes from calculating the **weighted mean** of all of the points with the same transient memberships.
     - the weighted mean does not only account for the parameters of each pt, but also account for how much it belongs.
   - > the `new var` for cluster_A, given the result of step_02(transient memberships), comes from calculating the **weighter VAR** of all of the points with the same transient memberships.       
<img src="https://user-images.githubusercontent.com/31917400/41906886-16dae142-7937-11e8-975d-e07f990de95f.jpg" />

 - Step_04. Compare(overlay) the new result with the old Gaussian. We iterate these steps until it converges(no movement?).  
   - > Evaluate the `log-likelihood` which sums for all clusters.
     - the higher the value, the more sure we are that the mixer model fits out dataset.
     - the purpose is to **maximize** this value by choosing the parameters(the mixing coefficient, mean, var) of each Gaussian again and again until the value converges, reaching a maximum. 
     - What's the mixing coefficient??????
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

__1.External Indices:__
<img src="https://user-images.githubusercontent.com/31917400/41922576-5570dfee-795d-11e8-8834-2fb0cda805b1.jpg" />

 - ARI(Adjusted Rand_Index) [-1 to 1]: 
<img src="https://user-images.githubusercontent.com/31917400/41938150-29f005e8-798a-11e8-8bac-85d9c0a0c0da.jpg" />




















































































