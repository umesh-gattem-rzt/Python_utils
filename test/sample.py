import os

import tensorflow as tf

logs_path = 'logs'

train_data = [1, 2, 3, 4, 5]
label = [6, 7, 8, 9, 10]

epoch_acc = tf.Variable(1)

tf.summary.scalar("acc", epoch_acc)

summary_op = tf.summary.merge_all()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    for i in range(5):
        # tf.add(12,13)

        _, summary = sess.run([epoch_acc, summary_op])
        # summary_writer.add_summary(summary, i)
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    summary_writer.close()
os.system("tensorboard --logdir=" + logs_path)
