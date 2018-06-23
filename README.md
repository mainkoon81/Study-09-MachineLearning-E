# Study-09-MachineLearning-E
UnsupervisedLearning

----------------------------------------------------------------------------------------------------------------------------------------
## Clustering
<img src="https://user-images.githubusercontent.com/31917400/41802151-ac0d97dc-7676-11e8-8c9f-30623f45fbbe.jpg" />

### 1. K-mean Clustering
 - Find the groups of similar observations
   - Step_01: randomly generate the centroids (k, k,...).
   - Step_02: Allocation
     - Holding 'k' fixed, label each data points (which k is closest?) and create clusters around each k (find the membership 'Z_ik') that **minimize SS**.
   - Step_03: Updating
     - Holding 'Z_ik' fixed, elect 'new k' for each cluster that **minimize SS**.
   - Step_04: Iterate until they converge (no movement of points b/w clusters)
> **SS** of each cluster..(find the membership `Z_ik` that minimize the SS)

![first](http://www.sciweavers.org/upload/Tex2Img_1529748773/render.png)

> **SS** for each data pt..(find the `MU_k` that minimize the SS)

![sec](http://www.sciweavers.org/upload/Tex2Img_1529755147/render.png)
































































































