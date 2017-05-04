import os

import tensorflow as tf

logs_path = 'logs'

train_data = [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7], [4, 5, 6, 7, 8, 9], [5, 6, 7, 8, 9]]

epoch_acc = tf.Variable(int(sum(train_data[0]) / len(train_data[0])))

tf.summary.scalar("acc", epoch_acc)

init = tf.global_variables_initializer()

summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter(logs_path)
    for i in range(5):
        summary = sess.run(summary_op)
        y = epoch_acc.assign(int(sum(train_data[i]) / len(train_data[i])))
        print(sess.run(y))
        summary_writer.add_summary(summary, i)
    summary_writer.close()
os.system("tensorboard --logdir=" + logs_path)
