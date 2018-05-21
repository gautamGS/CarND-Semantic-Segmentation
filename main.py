import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

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
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    #load model from the path provided
    tf.saved_model.loader.load(sess,[vgg_tag],vgg_path)
    
    #get graph of the loaded model
    savedGraph = tf.get_default_graph()
    
    #get tensors for input , keep_prob ,layre3 output , layer4 output , layer7 output
    image_input= savedGraph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob=savedGraph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out=savedGraph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out=savedGraph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out=savedGraph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out

tests.test_load_vgg(load_vgg, tf)

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    
    #adjust the depth of the layer 7 output to required final classes
    conv_1x1_layer7 = tf.layers.conv2d(vgg_layer7_out,num_classes,1,padding='SAME'
                                       ,kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),name="modified_conv_1x1_layer7")
    
    #upsample 2x times to the above layer (from x/32 to x/16)
    out_conv_1x1_layer7= tf.layers.conv2d_transpose(conv_1x1_layer7,num_classes,kernel_size=4,strides=(2,2),padding='SAME',
                                                    kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),name="modified_out_conv_1x1_layer7")
                                                       
    #use skip connection to add layer 4 output with above
    #but make sure to have same depth by using a 1x1 conv
    
    conv_1x1_layer4 = tf.layers.conv2d(vgg_layer4_out,num_classes,1,padding='SAME',
                                       kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),name="modified_conv_1x1_layer4")
    add_lay4_out_conv_1x1_layer7= tf.add(out_conv_1x1_layer7,conv_1x1_layer4,name="modified_add_lay4_out_conv_1x1_layer7")
    
    #upsample 2x times to the above layer (from x/16 to x/8)
    out_add_layer_7_4= tf.layers.conv2d_transpose(add_lay4_out_conv_1x1_layer7,num_classes,kernel_size=4,strides=(2,2),padding='SAME',
                                                  kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),name="modified_out_add_layer_7_4")
    
    #use skip connection to add layer 3 output with above
    #but make sure to have same depth by using a 1x1 conv
    conv_1x1_layer3 = tf.layers.conv2d(vgg_layer3_out,num_classes,1,padding='SAME',
                                       kernel_initializer= tf.random_normal_initializer(stddev=0.01)
                                       ,kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),name="modified_conv_1x1_layer3")
    add_lay3_out_add_layer_7_4= tf.add(out_add_layer_7_4,conv_1x1_layer3,name="modified_add_lay3_out_add_layer_7_4")
    
    ##upsample 8x times to the above layer (from x/8 to x) 
    
    lastLayer = tf.layers.conv2d_transpose(add_lay3_out_add_layer_7_4, num_classes,kernel_size=16,strides=(8,8),padding='SAME',
                                           kernel_initializer= tf.random_normal_initializer(stddev=0.01)
                                           ,kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),name="modified_lastLayer")
    
    
    
    return lastLayer

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
 
    #reshape from 4d to 2d for ease in classification
    logits = tf.reshape(nn_last_layer, (-1, num_classes),name="modified_logits")
    correct_label = tf.reshape(correct_label, (-1,num_classes),name="modified_correct_label")
    
    #apply cross entrophy
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=correct_label)
    
    cross_entropy_loss = tf.reduce_mean(cross_entropy)

    #add adam optimizer
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
                                 
    
    train_op = opt.minimize(
                     cross_entropy_loss, name="train_op"
                  )

    
    return logits, train_op, cross_entropy_loss


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
   
    sess.run(tf.global_variables_initializer())
    
 
    for epoch in range(epochs):
        lossV=0.0000
        count=0
        for X_batch , y_batch in get_batches_fn(batch_size):
            count+=1
            loss, _ = sess.run([cross_entropy_loss, train_op], feed_dict={
                input_image: X_batch,
                correct_label: y_batch,
                keep_prob: 0.8 
            })
            lossV +=loss
        
        print ("EPOCH : " ,epoch, " Loss : " ,lossV/count ) 

 
        

tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)
    epochs = 40
    batch_size=5
    learning_rate=0.001
    keep_prob=0.8

    
    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)
   
    
    with tf.Session() as sess:
       
    # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # Build NN using load_vgg, layers, and optimize function
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='modified_correct_label')
       
        input_layer, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        
        lastlayerReturned = layers(layer3, layer4, layer7, num_classes)
        
        logits, train_op, cross_entropy_loss= optimize(lastlayerReturned, correct_label, learning_rate, num_classes)
       
        print (" Performing Training for Epochs : ",epochs, " batch size : " ,batch_size , " learning rate : ",learning_rate  )
        
        # Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_layer,correct_label, keep_prob, learning_rate)    
        
        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_layer)
        
if __name__ == '__main__':
    run()