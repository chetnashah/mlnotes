
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


### TensorFlow

In tensorflow, data isn't stored as
integers, floats or strings, These values
are encapsulated in an object called a tensor.

Some of the constructs used to create such values are `tf.constant()`, `tf.variable()` etc.

Tensorflow's API is built around idea of a computational graph. The session is in charge of allocating operations to gpu or cpu including remote machines.
Run tensor(s) using `sess.run(tensor)`

#### Tesnorflow Input

Use `tf.placeholder()` to return a tensor that gets its value form data passed to `tf.session.run()` function, allowing you to set the input right before session runs.

Use `feed_dict` parameter in `tf.session.run()` to set placeholder tensor value.

``` py
x = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)

with tf.Session() as sess:
    output = sess.run(x, feed_dict={
        x: 'Test string',
        y: 12,
        z: 1.1
    })

// outputs 'Test string'
```

### Tensorflow operations

Operations like `tf.add()`, `tf.substract`,
`tf.multiply` etc. take in tensors and
return tensors.

Other commonly used functions are 
`tf.abs`, `tf.sign`, `tf.square`, `tf.sqrt`, `tf.pow`, `tf.exp`, `tf.log`, `tf.alltrigrunctions` etc. 

**Note** There may be typing issues, e.g. sqrt explects a float tensor, not an int tensor. Add substract etc work on tensor of same types e.g.
``` py
tf.subtract(tf.constant(2.0),tf.constant(1))  # Fails with ValueError: Tensor conversion requested dtype float32 for Tensor with dtype int32:
```

So in order to cast, you use `tf.cast` and specify it value and type you want to cast it to.

e.g.
``` py
tf.subtract(tf.cast(tf.constant(2.0), tf.int32), tf.constant(1))   # 1
```

Here is how you do:s 10/2 - 1
``` py
import tensorflow as tf

x = tf.constant(10)
y = tf.constant(2)
z = tf.subtract(tf.cast(tf.divide(x,y), tf.int32), 1)

with tf.Session() as sess:
    print(sess.run(z))

```
