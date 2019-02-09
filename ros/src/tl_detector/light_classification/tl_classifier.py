from styx_msgs.msg import TrafficLight
import numpy as np
import tensorflow as tf
import time
import rospy

'''
Source codes:
https://github.com/googlecodelabs/tensorflow-for-poets-2/blob/master/scripts/label_image.py
'''

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.model_file = 'light_classification/retrained_graph.pb'
        self.graph = self.load_graph(self.model_file)
        self.labels = ['green', 'none', 'red', 'yellow']
        self.light_states = [TrafficLight.GREEN, TrafficLight.UNKNOWN, TrafficLight.RED, TrafficLight.YELLOW]

    def load_graph(self, model_file):
        graph = tf.Graph()
        graph_def = tf.GraphDef()

        with open(model_file, "rb") as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)
        return graph

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        input_height=244
        input_width=244
        input_mean=0
        input_std=255
        input_name = "file_reader"
        output_name = "normalized"

        input_operation = self.graph.get_operation_by_name('import/input');
        output_operation = self.graph.get_operation_by_name('import/final_result');

        with tf.Session(graph=self.graph, config=tf.ConfigProto(log_device_placement=True)) as sess:
            start = time.time()
            file_reader = tf.read_file(image, input_name)
            image_reader = tf.image.decode_jpeg(file_reader, channels = 3, name='jpeg_reader')
            float_caster = tf.cast(image_reader, tf.float32)
            dims_expander = tf.expand_dims(float_caster, 0);
            resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
            normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
            tensor = sess.run(normalized)
            results = sess.run(output_operation.outputs[0], {input_operation.outputs[0]: tensor})
            end=time.time()
            #rospy.logwarn('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))

        results = np.squeeze(results)
        top_k = results.argsort()[-5:][::-1]
        label_idx = top_k[0]
        template = "{} (score={:0.5f})"
        for i in top_k:
            rospy.logwarn(template.format(self.labels[i], results[i]))
        return self.light_states[label_idx]
