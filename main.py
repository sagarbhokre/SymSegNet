import sys, os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('mode', "train", "Mode train/test")
tf.flags.DEFINE_integer("epochs", "20", "Number of epochs to run the optimizer")
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_float("learning_rate", "0.00001", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("logs_dir", "./logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "./data/", "path to dataset")
tf.flags.DEFINE_string("runs_dir", "./runs/", "path to store output files")
tf.flags.DEFINE_string("checkpoint_dir", "./chkpt/", "Path to checkpoint")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)

    image_input = tf.get_default_graph().get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = tf.get_default_graph().get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = tf.get_default_graph().get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = tf.get_default_graph().get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = tf.get_default_graph().get_tensor_by_name(vgg_layer7_out_tensor_name)

    DISPLAY_LAYERS = False
    if DISPLAY_LAYERS:
        print ("Path: ", vgg_path, os.path.isfile(vgg_path))
        # https://github.com/AKSHAYUBHAT/VisualSearchServer/blob/master/notebooks/notebook_network.ipynb
        for operation in tf.get_default_graph().get_operations():
            if "save" in operation.name:
                continue
            print ("Operation:",operation.name)
            for k in operation.inputs:
                print (operation.name,"Input ",k.name,k.get_shape())
            for k in operation.outputs:
                print (operation.name,"Output ",k.name)
            print ("\n")

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    x = vgg_layer7_out;

    x = tf.layers.conv2d_transpose(x, 512, [3,3], [2,2], padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv_d4')
    x = tf.add(x, vgg_layer4_out, name="fuse_i1")
    x = tf.layers.conv2d_transpose(x, 256, [3,3], [2,2], padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv_d3')
    x = tf.add(x, vgg_layer3_out, name="fuse_i2")
    x = tf.layers.conv2d_transpose(x, 128, [3,3], [2,2], padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv_d2')
    x = tf.layers.conv2d_transpose(x, 64, [3,3], [2,2], padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv_d1')
    x = tf.layers.conv2d_transpose(x, num_classes, [3,3], [2,2], padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv_d0')

    DISPLAY_LAYERS = False
    if DISPLAY_LAYERS:
        # https://github.com/AKSHAYUBHAT/VisualSearchServer/blob/master/notebooks/notebook_network.ipynb
        for operation in tf.get_default_graph().get_operations():
            print ("Operation:",operation.name)
            for k in operation.inputs:
                print (operation.name,"Input ",k.name,k.get_shape())
            for k in operation.outputs:
                print (operation.name,"Output ",k.name)
            print ("\n")

    return x
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
  
    #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_loss)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss) 
    return logits, train_step, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    num_classes = 2
    for epoch in range(epochs):
        n_batch = 0
        max_cel = 0.0
        for batch_xs, batch_ys in get_batches_fn(batch_size):
            #iou, iou_op = tf.metrics.mean_iou(out_layer, correct_label, num_classes)
            #print (sess.run(iou_op))
            n_batch += 1
            if batch_ys.shape[0] == batch_size:
                _, cel = sess.run([train_op, cross_entropy_loss], feed_dict={input_image: batch_xs,
                                                                           correct_label: batch_ys,
                                                                           keep_prob: 0.7,
                                                                           learning_rate: FLAGS.learning_rate})
                print ("Epoch: {}, Batch: {}, Cross entropy loss: {}".format(epoch, n_batch, cel))
                if cel > max_cel:
                    max_cel = cel
            #else:
            #    print("Skip files: ", batch_ys.shape[0])

        # Early exit of loss is acceptable
        if(max_cel <= 0.005):
            return True
    
    return False
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    tests.test_for_kitti_dataset(FLAGS.data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(FLAGS.data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(FLAGS.data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(FLAGS.data_dir, 'data_road/training'), image_shape)

        correct_label = tf.placeholder(tf.float32, [FLAGS.batch_size, image_shape[0], image_shape[1], num_classes], name="gt_ph")
        learning_rate = tf.placeholder(tf.float32, shape=(), name="lr_ph")

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)

        nn_last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)

        #print(sess.run(tf.report_uninitialized_variables()))
        if FLAGS.debug:
            train_writer = tf.summary.FileWriter(FLAGS.logs_dir, tf.get_default_graph())
            train_writer.add_graph(sess.graph)
            train_writer.close()
        
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restoring model from " + FLAGS.checkpoint_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("no checkpoint found")

        if FLAGS.mode == 'train':
            print ("Train mode");
            # Train NN using the train_nn function
            save_flag = train_nn(sess, FLAGS.epochs, FLAGS.batch_size, get_batches_fn, train_op, cross_entropy_loss, image_input,
                                    correct_label, keep_prob, learning_rate)

            if len(sys.argv) > 1 and "save" in sys.argv[1]:
                save_flag = True

            if save_flag:
                saver.save(sess, FLAGS.checkpoint_dir + 'model.ckpt', global_step=1)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(FLAGS.runs_dir, FLAGS.data_dir, sess, image_shape, logits, keep_prob, image_input)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
