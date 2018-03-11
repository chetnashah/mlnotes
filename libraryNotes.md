
### Numpy

#### Reading shapes

The shape `(45L)` simply means an array with 45 elements.

The shape `(2, 45)` simply means an array of two arrays of 45 elements each

The shape `(10, 2, 45)` simply means an array of ten arrays of two arrays with 45 elements each.

Normally we deal with 2-d shapes for csvs and tables i.e
The numpy shape `(4500, 4)` means 4500 rows/instances with 4 features for each instance/row.

* numpy.random.rand takes in a shape and returns all random numbers in that shape (uniformly in 0-1)

* Always have shapes with both `(x,y)` for vectors e.g. for row vector use `(1, 7)`. For column vector use `(5, 1)` for behaving consistently.

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

# place holders are usually used for
# inputs and supplied via feed_dict

with tf.Session() as sess:
    output = sess.run(x, feed_dict={
        x: 'Test string',
        y: 12,
        z: 1.1
    })

// outputs 'Test string'
```

### Tensorflow variables

Tensors whose value have to be changed, for e.g. weights etc. must be made using `tf.Variable()`.
A TensorFlow variable is the best way to represent shared, persistent state manipulated by your program.

When you launch the graph, variables have to be explicitly initialized before you can run Ops that use their value.
Initialization for tf.Variable:
``` py
b = tf.Variable(tf.zeros(1,5))
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
```

Another example:
``` py
W1 = tf.ones((2,2))
W2 = tf.Variable(tf.zeros((2,2)), name="weights")
with tf.Session() as sess:
    print(sess.run(W1)) #W1 does not need variables
    sess.run(tf.global_variables_initializer())
    print(sess.run(W2))
```

Initializing the weights with random numbers from a normal distribution is good practice

Other way to make variables is to use `tf.get_variable("vname", vvalue)`
e.g.
``` py
my_variable = tf.get_variable("my_variable", [1, 2, 3])
```

``` py
v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
w = v + 1  # w is a tf.Tensor which is computed based on the value of v.
           # Any time a variable is used in an expression it gets automatically
           # converted to a tf.Tensor representing its value.
```

#### updating tf Variables

To assign a value to a variable, use the methods assign, assign_add, and friends in the tf.Variable class. For example, here is how you can call these methods:

``` py
v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
assignment = v.assign_add(1)
tf.global_variables_initializer().run()
sess.run(assignment)  # or assignment.op.run(), or assignment.eval()
```

### Tensorflow Tensor Class

Represents one of the outputs of an Operation.

A Tensor is a symbolic handle to one of the outputs of an Operation. (can be run via session).

A Tensor can be passed as an input to another Operation. This builds a dataflow connection between operations

### Tensorflow operations

Represents a graph node that performs computation on tensors.

An Operation is a node in a TensorFlow Graph that takes zero or more Tensor objects as input, and produces zero or more Tensor objects as output.

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

`np.dot(a,b)` is same as `tf.matmul(a,b)` in tf
`a.shape` is same as `a.get_shape()` in tf
