
### Numpy

#### Reading shapes

The shape `(45L)` simply means an array with 45 elements.

The shape `(2, 45)` simply means an array of two arrays of 45 elements each

The shape `(10, 2, 45)` simply means an array of ten arrays of two arrays with 45 elements each.

Normally we deal with 2-d shapes for csvs and tables i.e
The numpy shape `(4500, 4)` means 4500 rows/instances with 4 features for each instance/row.

* numpy.random.rand takes in a shape and returns all random numbers in that shape (uniformly in 0-1)


#### Sklearn Interface design

* Estimators - Can estimate parameters given a dataset. Have a `fit()` method. Usually have one parameter i.e. (data), two parameters for supervised learning (data, labels, [hyperparameters]) with optional hyperparameters.

* Transformers - Some estimators can also transforma dataset, so they are known as trasnformers. Have a `.fit()` method, a `.transform()` and usually a `.fit_transform()` method.

* Predictors - Some estimators are capable of making predictions given a
dataset, they are called predictors. A predictor has a `.predict(data)` method that takes a dataset of new instances and returns dataset of corresponding predictions. It also has a `.score()` method to determine quality of predictions.


### Preprocessing in sklearn

* LabelBinarizer does one-hot encoding.


