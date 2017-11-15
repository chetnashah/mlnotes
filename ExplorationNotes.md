
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


