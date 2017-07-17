import tensorflow as tf
import numpy as np

import time

# this is the version that computes length of interaction for only one voxel at a time
# if speed is promising, to be used for actual TOR computations with random points in a cylindrical region
class rtrace():

    def __init__(self, num_points_py):
        self.build_graph(num_points_py)

    def build_graph(self, num_points_py):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # first, put all points into one "active" voxel by modulo-division
            self.points = tf.placeholder(tf.float32, shape = (num_points_py, 3), name = 'points')
            self.voxel_size = tf.placeholder(tf.float32, shape = (3), name = 'voxel_size')
            self.ray_vec = tf.placeholder(tf.float32, shape = (num_points_py, 3), name = 'ray_vec') # must be normalized!
            self.norm_const = tf.placeholder(tf.float32, (num_points_py), name = 'norm_const')

            self.ray_vec_us = tf.unstack(self.ray_vec, axis = 1)

            # this is where the center of the active voxel sits
            self.center = tf.divide(self.voxel_size, tf.constant(2.0))
            self.center_us = tf.unstack(self.center, axis = 0)

            # get the voxel indices in which each of these points lie
            self.voxel_ind = tf.floordiv(self.points, self.voxel_size)

            # and use these to compute the fractional part (i.e. within each voxel) of the point coordinates
            self.points_contracted = tf.multiply(self.voxel_ind, self.voxel_size)
            self.points_contracted = tf.subtract(self.points, self.points_contracted)

            # then unstack them for easy access to x / y / z - components (needed for computing the signed distance function)
            self.points_contracted_us = tf.unstack(self.points_contracted, axis = 1)

            # adds a new evaluation of the signed distance function to the TF graph
            def add_sdf(x, y, z):
                tmp_x = tf.subtract(tf.abs(tf.subtract(x, self.center_us[0])), self.center_us[0])
                tmp_y = tf.subtract(tf.abs(tf.subtract(y, self.center_us[1])), self.center_us[1])
                tmp_z = tf.subtract(tf.abs(tf.subtract(z, self.center_us[2])), self.center_us[2])
                
                tmp_max = tf.maximum(tmp_x, tmp_y)

                return tf.maximum(tmp_max, tmp_z)

            # body of the iteration loop (pass x / y / z coordinates through the iterations in separate streams to avoid repeated stacking / unstacking
            def body(t_fw, t_bw, i):
                # compute the current position to evaluate the SDF at
                pts_fw_x = tf.add(self.points_contracted_us[0], tf.multiply(t_fw, self.ray_vec_us[0]))
                pts_fw_y = tf.add(self.points_contracted_us[1], tf.multiply(t_fw, self.ray_vec_us[1]))
                pts_fw_z = tf.add(self.points_contracted_us[2], tf.multiply(t_fw, self.ray_vec_us[2]))

                pts_bw_x = tf.subtract(self.points_contracted_us[0], tf.multiply(t_bw, self.ray_vec_us[0]))
                pts_bw_y = tf.subtract(self.points_contracted_us[1], tf.multiply(t_bw, self.ray_vec_us[1]))
                pts_bw_z = tf.subtract(self.points_contracted_us[2], tf.multiply(t_bw, self.ray_vec_us[2]))

                # evaluate the SDF at the current positions
                sdf_fw = add_sdf(pts_fw_x, pts_fw_y, pts_fw_z)
                sdf_bw = add_sdf(pts_bw_x, pts_bw_y, pts_bw_z)

                # need to subtract instead of add since SDF is negative on the inside of the voxel, where we *always* are
                t_fw_new = tf.subtract(t_fw, tf.minimum(sdf_fw, tf.constant(-0.0001)))
                t_bw_new = tf.subtract(t_bw, tf.minimum(sdf_bw, tf.constant(-0.0001)))

                # return the updated points and t-values
                return (t_fw_new, t_bw_new, tf.subtract(i, 1))

            def cond(t_fw, t_bw, i):
                return tf.greater(i, 0)

            self.t_cur_fw = tf.zeros(num_points_py)
            self.t_cur_bw = tf.zeros(num_points_py)

            self.i = tf.constant(2)
            self.loop = tf.while_loop(cond, body, (self.t_cur_fw, self.t_cur_bw, self.i))
           
            # finally, add the two measured distances to get the actual length of interaction the LOR spends in this voxel. Multiply with the normalization constant
            self.loi = tf.multiply(self.norm_const, tf.add(self.loop[0], self.loop[1]))
            self.loi = tf.expand_dims(self.loi, axis = 1)

            # for the output, concat the voxel indices with the LOIs
            self.out = tf.concat([self.loi, self.voxel_ind], axis = 1, name = 'out')

    def start_tf_session(self):
        self.settings = tf.ConfigProto(device_count = {'GPU': 0})
        self.sess = tf.Session(graph = self.graph, config = self.settings)

    def stop_tf_session(self):
        self.sess.close()

    def run(self, points_in, ray_vec_in, voxel_size_in, norm_const_in):
        # first, given the start and end points of the LOR, generate the points through which line segments are traced
        # for the time being, choose them to lie directly on the line itself
        # later, if performance is promising, could extend it to automatically sample a 3D TOR from the very beginning

        # actually perform the ray tracing (actually, ray marching)
        result = self.sess.run(self.out, feed_dict = {self.points: points_in, self.voxel_size: voxel_size_in, self.ray_vec: ray_vec_in, self.norm_const: norm_const_in})
        return result

    def save_graph_to_file(self, filename):
        tf.train.write_graph(self.sess.graph_def, '.', filename, as_text = False)

# main program:
voxel_size = np.array([1.0, 1.0, 1.0])

# prepare a list of points
testpoints = np.array([[1.5, -0.5, 0.5], [-0.1, 0.5, 1.5]])

testvec = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 1.0]]) # along space diagonal
norms = np.expand_dims(np.linalg.norm(testvec, axis = 1), axis = 1)
testvec = np.divide(testvec, norms)

norm_const = np.array([1.0, 1.0])

number_points = 26000000
#number_points = 20000

rtr = rtrace(number_points)
rtr.start_tf_session()

rtr.save_graph_to_file('TFRayMarching.pb')

rtr.stop_tf_session()
