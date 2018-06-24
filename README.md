# Study-09-MachineLearning-E
UnsupervisedLearning

----------------------------------------------------------------------------------------------------------------------------------------
## Clustering
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
Sometimes some column has smaller values than the rest of the columns, and so its variance counts for less in the clustering process (since clustering is based on distance). We normalize the dataset so that each dimension lies between 0 and 1, so they have equal weight in the clustering process. **This is done by subtracting the minimum from each column then dividing the difference by the range.** Would clustering the dataset after this transformation lead to a better clustering?
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
































































































