import tensorflow as tf


xc1 = tf.constant([35, 40, 45], name='xc1')
xc2 = tf.constant([5, 10, 20], name='xc2')
yv = tf.Variable(tf.add(xc1, xc2), name='yv') # variable initialized not update

update_op = tf.assign(yv, tf.add(yv, tf.constant([11,11,11])))

init_op = tf.global_variables_initializer()

with tf.Session() as session:
      # `sess.graph` provides access to the graph used in a `tf.Session`.
    writer = tf.summary.FileWriter("tfdump/log", session.graph)
    session.run(init_op)
    print(session.run(yv))
    print(session.run(update_op))
    writer.close()
