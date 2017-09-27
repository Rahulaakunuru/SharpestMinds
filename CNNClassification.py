import tensorflow as tf
import math
import ImageRead
from timeit import default_timer as timer

(images, bounding_box, labels, img_width, img_height) = ImageRead.getDataSet()
x_train, x_test, y_bounding_box_train, y_bounding_box_test, y_classification_train, y_classification_test = ImageRead.getTrainTestData(images, bounding_box, labels, test_size = 0.998)
learning_rate = 0.05
epochs = 5
batch_size = 1

x_shaped = tf.placeholder(tf.float32, [None,img_width,img_height,1])
y_classification = tf.placeholder(tf.float32, [None,10])
y_box = tf.placeholder(tf.float32, [None,4])


def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, name, req_pool = True, conv_strides = [1,1,1,1], 
		pool_strides = [1,2,2,1]):
	conv_filts_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]
	weights = tf.Variable(tf.truncated_normal(conv_filts_shape,stddev = 0.03), name = name+'_w')
	bias = tf.Variable(tf.truncated_normal([num_filters]), name = name+'_b')
	#Filter format is [filter_height, filter_width, in_channels, out_channels]
	out_layer = tf.nn.conv2d(input_data,weights,conv_strides, padding='SAME')
	out_layer += bias
	out_layer = tf.nn.relu(out_layer)
	if req_pool:
		ksize = [1,2,2,1]
		out_layer = tf.nn.max_pool(out_layer,ksize=ksize,strides=pool_strides,padding='SAME')
	return out_layer

layer1 = create_new_conv_layer(x_shaped,1,128,[11,11],name='layer1', conv_strides = [1,4,4,1])
layer2 = create_new_conv_layer(layer1,128,256,[5,5],name='layer2')
layer3 = create_new_conv_layer(layer2,256,512,[3,3],name='layer3')
layer4 = create_new_conv_layer(layer3,512,1024,[3,3],name='layer4')
layer5 = create_new_conv_layer(layer4,1024,1024,[3,3],name='layer5')

flattened_size = int(math.ceil(img_height/128.0)*math.ceil(img_width/128.0)*1024)
flattened = tf.reshape(layer5,[-1,flattened_size])

wb1 = tf.Variable(tf.truncated_normal([flattened_size,1000],stddev=0.03),name='wb1')
bd1 = tf.Variable(tf.truncated_normal([1000], stddev = 0.01),name='wb1')
dense_layer1 = tf.matmul(flattened,wb1) + bd1
dense_layer1 = tf.nn.relu(dense_layer1)

"""
Classification Head
"""
wb2 = tf.Variable(tf.truncated_normal([1000,10],stddev=0.03),name='wb2')
bd2 = tf.Variable(tf.truncated_normal([10],stddev=0.01),name='bd2')
dense_layer2 = tf.add(tf.matmul(dense_layer1,wb2),bd2)

y_ = tf.nn.softmax(dense_layer2)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=y_classification))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_classification,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

"""
Regression Head for Image Localization
"""
wb_reg2 = tf.Variable(tf.truncated_normal([1000,4],stddev = 0.03), name = 'wb_reg2')
bd_reg2 = tf.Variable(tf.truncated_normal([4],stddev = 0.01), name = 'bd_reg2')
regression_dense_layer = tf.add(tf.matmul(dense_layer1, wb_reg2), bd_reg2)

cost_l2 = tf.reduce_mean(tf.square(regression_dense_layer - y_box))
optimizer_l2 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_l2)

start = timer()
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init_op)
	total_batch = int(len(x_train)/batch_size)
	for epoch in range(epochs):
		avg_cost = 0
		for i in range(total_batch):
			batch_x,batch_y_box,batch_y_classification = ImageRead.getNextBatch(x_train, y_bounding_box_train, y_classification_train, i*batch_size, batch_size)
			_,cost_classification,__,cost_regression = sess.run([optimizer,cross_entropy,optimizer_l2,cost_l2], feed_dict={x_shaped:batch_x, y_classification:batch_y_classification,y_box:batch_y_box})
			avg_cost+=(cost_classification+cost_regression)/batch_size
		#test_acc = sess.run(accuracy,feed_dict={x_shaped:x_test, y_classification:y_classification_test})
		#print 'Epoch: {} Cost: {:.3f} Test Accuracy: {:.3f}'.format(epoch+1,avg_cost,test_acc)
		print 'Epoch: {} Cost: {:.3f}'.format(epoch+1,avg_cost)
	print '\nTraining Complete!!!'
	end = timer()
	print 'Time taken to complete the training {0}'.format(end-start)
	#print sess.run(accuracy,feed_dict={x_shaped:x_test, y_classification:y_classification_test})