
### Statistical Exploration 

* Try to articulate what it is you are measuring and how it is that 
you are measuring.

For a given expirement it is recommended to have
1. a good sample size.
2. A representative sample.
3. A sound methodology.

External factors(lurking variables) should be kept constant
accross all test subjects as possible so that they might not influence
result.

Average of population is denoted by mu, and average of sample statistic is denoted by \bar{x}

* To show correlation, observations are enough.
* To show causation, you have to do a controlled experiment.

* Surveys have response bias and non-response bias (look this up)

Randomized assignment, double blind experiments should be used to prove causality. 

Histograms tend to have bins(interval of domain values) with frequencies, instead of 
each possible value in domain with it's individual frequencies (largely sparse).
Smaller the bin size, more detail you have about the data distribution.
Larger bin size sacrifice detail for convinience.

* Key fact: a random sample tends to exhibit same properties as the population from which it is drawn.

* As variance grows, we need larger samples to have same degree of confidence.

* Binomial distribution is really important for information theory.

### Exploration with Python

* parameters is what model predicts, hyper-parameters is what the programmer has to provide from outside, e.g. no of branching etc in 
decision trees, C value in SVM etc. For searching through the space
of possible hyper parameters GridSearch or RandomizedSearch is used
from sklearn.

* Using pandas/numpy/scikitlearn
    - Pandas has two basic structures : Series which is like a numpy array and Dataframe which is like a two d table or like an excel sheet.
    - Useful methods on pd.Series : s.describe(), s.value_counts(), s.hist(bins, etc)

* Use of random state in sklearn code:
It doesn't matter if the random_state is 0 or 1 or any other integer. What matters is that it should be set the same value, if you want to validate your processing over multiple runs of the code.

* Use seaborn boxplots, and learn how boxplots look for normal distributions and skewed distributions.

* Always use a scatter-matrix to get data insight. via 
```
// Extremly useful to find out correlation between any two variables
pd.scatter(dataframe,options)
```
It plots scatter plot for each combination of two variables, allowing you to see if there is correlation between any two pair of variables. In the diagonal it shows histogram for the particular variable.
If you see a lot of skewing, try again with log transformation.

* When there are no outliers in a sample, the mean and standard deviation are used to summarize a typical value and the variability in the sample, respectively.  When there are outliers in a sample, the median and interquartile range are used to summarize a typical value and the variability in the sample, respectively

* If data is not normally distributed(where usually mean and median nearby), especially if the mean and median vary significantly (indicating a large skew), it is most often appropriate to apply a non-linear scaling (like logarithm) — particularly for financial data.


* One of the known techniques to test feature relevance, is to remove that feature
and try to predict that feature from the rest of features, if it is high, it is possible that the removed feature is some sort of combination of remaining features meaning it is not relevant, where as if the score is low to predict removed feature means it is more relevant.

* visualizing with heatmaps = http://seaborn.pydata.org/generated/seaborn.heatmap.html


* Finding outliers in the dataset - http://datapigtechnologies.com/blog/index.php/highlighting-outliers-in-your-data-with-the-tukey-method/

* K Means is sensitive to outliers : The reason is simply that k-means tries to optimize the sum of squares. And thus a large deviation (such as of an outlier) gets a lot of weight, and pulls the centroids towards itself.

* General advice: Outlier removal should be done after deliberate and thorough checking. Rationale : 
- An outlier is an observation that appears to deviate markedly from other observations in the sample An outlier may indicate bad data. For example, the data may have been coded incorrectly or an experiment may not have been run correctly. 
- If it can be determined that an outlying point is in fact erroneous, then the outlying value should be deleted from the analysis (or corrected if possible).
- In some cases, it may not be possible to determine if an outlying point is bad data. Outliers may be due to random variation or may indicate something scientifically interesting. In any event, we should not  simply delete the outlying observation before a through investigation. In  running experimdnts , we may repeat the experiment. If the data contains significant outliers, we may need to consider the use of robust statistical techniques.

- You should proceed with caution when considering to remove observations from the data. In many cases, there is a valid reason for these observations to be outliers and that is what the researcher should be studying. Why was this an outlier?
- Another issue with outliers is where to draw the line. It may not be clear where the outlier behavior starts. There are some people who arbitrarily eliminate a percentage at the tails (e.g. 5%), which makes no sense whatsoever.
- Finally, you should not take out the outliers and then transform the data. The data may appear non-normally distributed because of those data points. So eliminating them may in fact cause the data to appear normally distributed. So by transforming the data, you didn't improve the fit.

* Another way of getting around outliers:
- If the number of outliers is small and you are concerned that they will destabilize your solution, you could attempt a random forest classifier. The RF fits trees to random selections of data and variables, and collects "votes" from each, thus reducing the impact of outlier valuers.

- On the other hand, if the number of outliers is fairly substantital, you might want to create a new class called "outlier". In the training set, apply this label to those values you have deemed to be outliers and then fit the model with the augmented class. Check if the model correctly identifies outliers in the test set.

* IQR (InterQuartile range) is Q3 - Q1.

* Turkey fences are (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

* Why are turkey fences preferred ? There are different methods to detect the outliers, including standard deviation approach and Tukey’s method which use interquartile (IQR) range approach. Tukey’s method is not dependent on distribution of data. Moreover, the Tukey’s method ignores the mean and standard deviation, which are influenced by the extreme values (outliers).

* **Mahalanobis distance** : Measure of distance between point P and distribution D. It is a multidimensional generalization of idea of measuring how many SDs away P is from mean of D. 
    - The distance is zero if P is at mean of D.
    - Along each principal component axis, it measures number of standard deviations from P to the mean of D.
    - Mahalanobis distance is unitless and scale-invariant(normalized euclidean distance) = (x - mu)/sigma.
    - Mahalanobis distance is preserved under full-rank linear transformations of the space spanned by the data. This means that if the data has a nontrivial nullspace, Mahalanobis distance can be computed after projecting the data (non-degenerately) down onto any space of the appropriate dimension for the data.

* pca.fit only learns variances, does not change original data

* pca.transform does actual compression/dimensionality reduction/compression.

* Many of the times dimensionality reduction/pca is done before doing clustering. Find out more about this why? If a significant amount of variance is explained by 2 or 3 features in explained_variance_Ratio of pca fitting, we should go ahead with pca compression/dimensionality reduction, to be able to visualize clustering easily later on.

* EM algorithm : https://www.youtube.com/watch?v=REypj2sy_5U&list=PLBv09BD7ez_4e9LtmK626Evn1ion6ynrt

#### Common scores and measures (Evaluation metrics)

In prediction, both False positives(FP) and false negatives(FN) are bad, but under different conditions, you
would want to minimize different one, or both. Find out more at (https://en.wikipedia.org/wiki/Confusion_matrix)

* Precision: Talks only about positives,
```
Precision = TP/(TP + FP)
```

* Recall:
```
Recall = TP/(TP + FN)
```

* Accuracy:
```
Accuracy = (TP + TN)/(FP + FN + TP + TN)
```

* F-1 score :
```
F1 score = 2 * precision * recall / (precision + recall)
```

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


#### Statistics Notes

* Simplest form of non-parametric density estimation is Histogram. Although it has two parameters, bin size and starting bin position.

* kde is improved (histogram) - method of estimating pdf of underlying dataset.

#### Markovian Property

The Markov property is used to refer to situations where the probabilities of different outcomes are not dependent on past states: the current state is all you need to know, sometimes called "memorylessness".


#### Markov Decision Process

Consists of SMAR - States, Model/Trasition table, Actions, Rewards(real valued).


## Neural nets

* Vectorization means getting rid of for loop and using more mathematical notation to get rid of index of the for-loop.

* To vectorize for multiple examples (usually denoted by m), stack each example horizontally to each other.

### Activation functions

* sigmoid : Real numbers -> [0,1] (only use on output layer for binary classification)
* tanh : Real numbers -> [-1, 1] (superior than sigmoid)

Problem with above activation is plateauing of output for large values.
And their derivative/slope is almost flat, thus not helping back-propogation/gradient descent to qucikly converge.

* ReLU: Rectified linear unit (It is non-linear btw). `output = max(0, input)`

* Why is some non-linearity needed in activation functions? Unless some non-linearity is present in activation function, all we are computing are linear functions of input, which cannot cover intersting hyperspaces that our target function actually is. Think of the X0R problem.

* What to use at output layer? Binary classification - sigmoid,
multilabel classification - softmax, value prediction - linear unit?

### derivative of activation functions are important!