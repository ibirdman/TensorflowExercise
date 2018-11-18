import tensorflow as tf
import matplotlib.image as img
import matplotlib.pyplot as plot
myfile = "likegeeks.png"
myimage = img.imread(myfile)
slice = tf.placeholder("int32",[None,None,3])
cropped = tf.slice(myimage,[10,0,0],[516,-1,-1])
sess = tf.Session()
result = sess.run(cropped, feed_dict={slice: myimage})
plot.imshow(result)
plot.show()
