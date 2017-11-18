
### Exploration with Python

* Using pandas/numpy/scikitlearn
    - Pandas has two basic structures : Series which is like a numpy array and Dataframe which is like a two d table or like an excel sheet.
    - Useful methods on pd.Series : s.describe(), s.value_counts(), s.hist(bins, etc)

* Always use a scatter-matrix to get data insight. via 
```
// Extremly useful to find out correlation between any two variables
pd.scatter(dataframe,options)
```
It plots scatter plot for each combination of two variables, allowing you to see if there is correlation between any two pair of variables. In the diagonal it shows histogram for the particular variable.
If you see a lot of skewing, try again with log transformation.

* If data is not normally distributed(where usually mean and median nearby), especially if the mean and median vary significantly (indicating a large skew), it is most often appropriate to apply a non-linear scaling (like logarithm) — particularly for financial data.

* One of the known techniques to test feature relevance, is to remove that feature
and try to predict that feature from the rest of features, if it is high, it is possible that the removed feature is some sort of combination of remaining features meaning it is not relevant, where as if the score is low to predict removed feature means it is more relevant.

* visualizing with heatmaps = http://seaborn.pydata.org/generated/seaborn.heatmap.html

* Finding outliers in the dataset - http://datapigtechnologies.com/blog/index.php/highlighting-outliers-in-your-data-with-the-tukey-method/

* pca.fit only learns variances, does not change original data

* pca.transform does actual compression/dimensionality reduction/compression.

* EM algorithm : https://www.youtube.com/watch?v=REypj2sy_5U&list=PLBv09BD7ez_4e9LtmK626Evn1ion6ynrt

### Clustering

#### Types of clustering

* monothetic vs polythetic clustering : In monothetic cluster members have some common property that you can pin down based on which cluster is formed(e.g. all males aged 20-35), where as in polythetic cluster, cluster members are made based on similarity measure but there is not a single common property that can be pinned down based on which cluster is formed.

* Soft vs hard clustering : 
    - Hard clustering: clusters don't overlap i.e. item belongs to a cluster or not. exclusive membership.
    - Soft clustering: Clusters may overlap, i.e. a belief, confidence, strength of association is given by item for each cluster it belongs to.

* Flat vs hierarchial clustering:
    - Flat clustring : set of groups/clusters
    - Hierarchial clustering : a taxonomy. i.e. cluster can contain sub-clusters and so on.

#### Clustering methods

* **K-D Trees** : tree like splitting method - monothetic(attribute based splitting), hard boundries, hierarchial

* **K means clutering** : split data into k subsets(user has to provide it) - polythetic, hard boundaries, flat
    - Converges to a local minimum(not global) and hence it should do multiple kmeans with different initial random centroid positions. The objective function for minimization is the aggregate intra-cluster distance(distances to centroid within a cluster).

    - Intersting observation is, the more clusters you add to the system, the lower the variance is going to be. Because for e.g. when K = n, variance will be 0. A good techinque to decide optimal number of cluster is to plot number of clusters vs variance  and decide the number of clusters where the variance drop slope stops becoming steep.

* **Gaussian Mixtures Model(EM algorithm)** : (Kmeans with soft boundaries) fits a mixture of K Gaussians to the data, like K means but probabilities are associated so soft boundaries, polythetic and flat.

* **Agglomerative clustering** : creates an ontology of nested sub-populations, polythetic, hard boundaries, hierarchial. (Don't know if it is same as single linkage clustering)

#### Evaluating cluster methods

* Extrinsic - how good it helps us solve another problem/classifier.

* Intrinsic - Is it good in and of itself. clusters correspond to classes.

* see silhoutte co-efficients in sklearn

#### Bayesian procedure

1. We choose a probability density p(theta) — called the prior distribution — that
expresses our beliefs about a parameter theta before we see any data.

2. We choose a statistical model p(x | theta) that reflects our beliefs about x given theta.
This is usally well known e.g. we decide parameters theta = some mu, sigma -> we get p(x|theta) by formula of pmf of gaussian distribution with mu and sigma. Usually p(x|theta) are easily calculatable by pmf/pdf formulas with theta.

3. After observing data Dn = {X1,...,Xn}, we update our beliefs and calculate
the posterior distribution p(theta | Dn). p(theta|Dn) is the real job of machine learning.


