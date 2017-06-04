import time
import os
import tensorflow as tf
from tensorflow.contrib import learn

from dataset import get_data_sets
import model as model

LOG_DIR = './output'
MAX_STEPS = 5000
BATCH_SIZE = 50


def placeholders(batch_size=100, img_shape=(64, 64, 3)):
    # these are the placeholders that will go into the graph where we will feed batches
    # of data during training and validation
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, *img_shape))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, labels_placeholder


def get_feed_dict(data_set, batch_size, images_pl, labels_pl):
    images_feed, labels_feed = data_set.next_batch(batch_size)
    feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
    }
    return feed_dict


def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set,
            batch_size):
    # evaluate the model by running through all of the images in the set and
    # calculating the fraction that were classified correctly
    true_count = 0
    num_epoch = data_set.num_examples // batch_size
    num_examples = num_epoch * batch_size
    for step in range(num_epoch):
        feed_dict = get_feed_dict(data_set, batch_size, images_placeholder, labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)

    precision = float(true_count) / num_examples
    print('  Num examples: %d  Num correct: %d  Precision: %0.04f' %
          (num_examples, true_count, precision))


def run_training(unused_argv):
    data_sets = get_data_sets('data/train/')

    with tf.Graph().as_default():
        # Generate placeholders for the images and labels that we feed into the graph.
        images_placeholder, labels_placeholder = placeholders(batch_size=BATCH_SIZE, img_shape=(64, 64, 3))

        # Build the graph that computes predictions for labels.
        # train_logits includes dropout to avoid overfitting. test_logits does not.
        train_logits = model.cnn_model(images_placeholder, mode='training')
        test_logits = model.cnn_model(images_placeholder, mode='test')

        # Add the Graph that computes the loss.
        loss = model.loss(train_logits, labels_placeholder)

        # Add the Graph the does the minimization of the loss.
        train_op = model.training(loss, learning_rate=0.1)

        # Add the graph to compare the predictions to the labels during evaluation.
        eval_correct = model.evaluate(test_logits, labels_placeholder)

        # Build the summary Tensor that can be visualized in TensorBoard.
        summary = tf.summary.merge_all()

        # Create an Op to initialize the variables
        init = tf.global_variables_initializer()

        # Create a saver for writing out training checkpoints.
        saver = tf.train.Saver()

        # Create a Tensor Flow "Session"
        sess = tf.Session()

        # Create a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

        # Initialize the variables.
        sess.run(init)

        # Start the training loop.
        for step in range(MAX_STEPS):
            start_time = time.time()

            # Fill a feed dictionary with the actual set of images and labels
            # for this particular training step.
            feed_dict = get_feed_dict(data_sets['train'], BATCH_SIZE, images_placeholder, labels_placeholder)

            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

            duration = time.time() - start_time

            # Write the summaries and print an overview to the console.
            if step % 10 == 0:
                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                # Update the events file.
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % 1000 == 0 or (step + 1) == MAX_STEPS:
                checkpoint_file = os.path.join(LOG_DIR, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
                # Evaluate against the training set.
                print('Training Data Eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets['train'],
                        batch_size=BATCH_SIZE)
                # Evaluate against the validation set.
                print('Validation Data Eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets['validation'],
                        batch_size=BATCH_SIZE)
                # Evaluate against the test set.
                print('Test Data Eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets['test'],
                        batch_size=BATCH_SIZE)


if __name__ == "__main__":
    tf.app.run(main=run_training)
