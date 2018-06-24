# Study-09-MachineLearning-E
UnsupervisedLearning

----------------------------------------------------------------------------------------------------------------------------------------
## Clustering
<img src="https://user-images.githubusercontent.com/31917400/41802151-ac0d97dc-7676-11e8-8c9f-30623f45fbbe.jpg" />

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
<img src="https://user-images.githubusercontent.com/31917400/41810519-d272fca6-76f7-11e8-97a0-abdef5c8d8ba.jpg" />

### 2. Hierarchical & Density-based Clustering
 - In SKLEARN, they are parts of 'agglomerative clustering' component.  
 - Hierarchical Clustering results in a **structure of clusters** that gives us a visual indication of how clusters relate to each other. 
 - DBSCAN(Density-based Spatial Clustering of Applications with Noise) clusters the pt densely packed together and labels other pt as noise. 
<img src="https://user-images.githubusercontent.com/31917400/41822691-a7ced232-77eb-11e8-946f-40b479b843be.jpg" />

> Hierarchical Clustering Example: A Pizza company want to cluster the locations of its customers in order to determine where it should open up its new branches.
 - Single-link clustering: 
   - Step01. assume each pt is already a cluster and we give each pt a label. 
   - Step02. calculate the distance b/w each pt and each other pt, then choose the smallest distances to group them into a cluster. On the side, we draw the structure tree one by one (the dendogram gives us an additional insight that might direct the results of the clustering misses) 
<img src="https://user-images.githubusercontent.com/31917400/41822846-fb1a5374-77ed-11e8-8f71-50aad55778a5.jpg" />































































































