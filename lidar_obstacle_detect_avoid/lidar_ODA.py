# BSD 3-Clause License
#
# Copyright (c) 2022 University of Wisconsin - Madison
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.#
###############################################################################
## Author: Harry Zhang
###############################################################################

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist, PoseStamped, TwistStamped
from chrono_ros_interfaces.msg import DriverInputs as VehicleInput
from chrono_ros_interfaces.msg import Body
from nav_msgs.msg import Path
from ament_index_python.packages import get_package_share_directory
import numpy as np
import random
import os
import torch
import torch.nn as nn
import csv 
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
import sys
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
ament_tools_root = os.path.join(os.path.dirname(__file__), '.')
sys.path.insert(0, os.path.abspath(ament_tools_root))
from ConditionalAvoidanceNN import NeuralNetwork
import time
import joblib

class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')

        # update frequency of this node
        self.freq = 10.0
        # declare parameters:
        self.declare_parameters(
            namespace='',
            parameters=[
                ('exp_index', 1, ParameterDescriptor(description='experiments index'))
            ]
        )
        self.exp_index = self.get_parameter('exp_index').get_parameter_value().integer_value
        # READ IN SHARE DIRECTORY LOCATION
        package_share_directory = get_package_share_directory('lidar_obstacle_detect_avoid')
        # initialize control inputs
        self.steering = 0.0
        self.throttle = 0.6
        self.braking = 0.0
        # vehicle power
        self.engine_speed = 0.0
        self.engine_tq = 0.0
        self.engine_speed_list = []
        self.engine_tq_list = []

        # initialize vehicle state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v = 0.0

        # data that will be used by this class
        self.state = Body()
        self.path = Path()
        self.go = False
        self.vehicle_cmd = VehicleInput()
        self.pc_data = []
        self.file = open("/sbel/Desktop/waypoints_paths/square.csv")
        self.ref_traj = np.loadtxt(self.file,delimiter=",")
        self.lookahead = 4.0
        self.counter = 1
        self.pc_input = []
        self.e_input = []
        self.pc_np = []
        self.aviodance_state = False
        self.avoid_right = True
        # publishers and subscribers
        qos_profile = QoSProfile(depth=1)
        qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        self.sub_state = self.create_subscription(PoseStamped, '/m113/state/pose', self.state_callback, qos_profile)
        self.sub_state = self.create_subscription(TwistStamped, '/m113/state/twist', self.vel_callback, qos_profile)
        self.pub_vehicle_cmd = self.create_publisher(VehicleInput, '/m113/driver_inputs', 5)
        self.sub_PCdata = self.create_subscription(PointCloud2,'/m113/pointcloud',self.lidar_callback,qos_profile)
        self.sub_Enginedata = self.create_subscription(Float64MultiArray,'/m113/engine_power',self.engine_callback,qos_profile)
        self.timer = self.create_timer(1/self.freq, self.pub_callback)
        self.bounding_box = []
        self.safety_coef = 0.01 + 0.61561513 # self defined safety value + threshold from the model training
        # self.avoid_model = NeuralNetwork()
        # self.avoid_model.load_state_dict(torch.load("/sbel/Desktop/ros_ws/src/lidar_obstacle_detect_avoid/lidar_obstacle_detect_avoid/tracked_veh_cond_avoid.pth"))
        # self.avoid_model.eval()
        self.scaler = joblib.load('/sbel/Desktop/ros_ws/src/lidar_obstacle_detect_avoid/lidar_obstacle_detect_avoid/scaler.pkl')
        self.avoid_model_sk = joblib.load('/sbel/Desktop/ros_ws/src/lidar_obstacle_detect_avoid/lidar_obstacle_detect_avoid/mlp_model.pkl')
        # self.get_logger().info('updated')
        
    def state_callback(self, msg):
        # self.get_logger().info("Received '%s'" % msg)
        self.state = msg
        self.x = msg.pose.position.x
        self.y = msg.pose.position.y
        #convert quaternion to euler angles
        e0 = msg.pose.orientation.x
        e1 = msg.pose.orientation.y
        e2 = msg.pose.orientation.z
        e3 = msg.pose.orientation.w
        self.theta = np.arctan2(2*(e0*e3+e1*e2),e0**2+e1**2-e2**2-e3**2)
    
    def engine_callback(self, msg):
        # Append new readings to the lists
        self.engine_speed_list.append(msg.data[0])
        self.engine_tq_list.append(msg.data[1])
        # Keep only the last 20 readings
        if len(self.engine_speed_list) > 5:
            self.engine_speed_list.pop(0)
        if len(self.engine_tq_list) > 5:
            self.engine_tq_list.pop(0)
        # Calculate the average
        self.engine_speed = sum(self.engine_speed_list) / len(self.engine_speed_list)
        self.engine_tq = sum(self.engine_tq_list) / len(self.engine_tq_list)
    
    def vel_callback(self, msg):
        self.v = np.sqrt(msg.twist.linear.x ** 2 + msg.twist.linear.y ** 2)
    
    def make_inference(self, model, new_features):
        # Convert new features to torch tensor
        new_features_tensor = torch.tensor(new_features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():  # No need to track gradients for inference
            output = model(new_features_tensor)
            predicted_prob = torch.sigmoid(output).item()
        return predicted_prob
    
    def Obstacle_Detection_ptClustering(self,eps=0.2, min_samples=20):
        points = self.pc_np
        # filter out the "ground" floor points
        # Prepare the features (x, y) and target (z)
        X = points[:, :2]  # x and y coordinates
        y = points[:, 2]   # z coordinate
        # Fit a plane using RANSAC
        ransac = RANSACRegressor()
        ransac.fit(X, y)
        # Identify inliers (ground) and outliers (non-ground)
        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)
        # Separate ground and non-ground points
        ground_points = points[inlier_mask]
        non_ground_points = points[outlier_mask]
        
        # Cluster the data
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(non_ground_points)
        labels = clustering.labels_
        unique_labels = set(labels)
        self.get_logger().info("unique: %s" % unique_labels)
        cluster_centers = []
        cluster_dim_array = []
        #print(unique_labels)
        self.bounding_box = np.zeros(3)
        for k in unique_labels:
            if k == -1:  # Skip noise points
                continue
            # Extract points belonging to the current cluster
            class_member_mask = (labels == k)
            cluster_points = non_ground_points[class_member_mask]
            
            # Check if cluster fits within the bounding box dimensions
            min_pt = np.min(cluster_points, axis=0)
            max_pt = np.max(cluster_points, axis=0)
            # compute bounding box dimension
            cluster_dim = max_pt - min_pt
            self.bounding_box = cluster_dim
            cluster_dim_array.append(cluster_dim)
            center = cluster_points.mean(axis=0)
            cluster_centers.append(center)
            
        # self.get_logger().info("bounding box: %s" % self.bounding_box)
        # obtain the closest point
        #self.get_logger().info("candidate obstacle position: %s" % cluster_centers)
        if  cluster_centers == [] or len(unique_labels)==1:
            self.get_logger().info("We good, don't see anything")
            self.aviodance_state = False
        else:
            # get the closest detected obstacle
            cluster_centers = np.array(cluster_centers)
            ind_closest = np.argmin( np.sum(cluster_centers[:, :2] ** 2, axis=1) )
            obs_center = cluster_centers[ind_closest]
            obs_dim = cluster_dim_array[ind_closest]
            
            self.get_logger().info("closest obstacle: %s" % obs_center)
            
            # clip_vel = np.clip(self.v,0.0,1.0)
            input_NN = np.array([[self.v, obs_dim[0], obs_dim[1], obs_dim[2],self.engine_tq*self.engine_speed]])
            input_NN = self.scaler.transform(input_NN)
            
            ############## could use different trained nn to predict the value##################
            #pre_vals = self.make_inference(self.avoid_model, input_NN)
            pre_vals = self.avoid_model_sk.predict_proba(input_NN)[:,1]
            self.get_logger().info('pre_value:'+str(pre_vals))
            # input_classifier = np.array([self.v,obs_dim[0], obs_dim[1], obs_dim[2],self.engine_tq*self.engine_speed]).reshape(1, -1)
            # safe_to_cross = predict_labels(self.avoid_classifier, self.avoid_scaler,input_classifier)
            
            if pre_vals < self.safety_coef:
            #if not safe_to_cross:
                self.get_logger().info("Identifier: Danger!!!")
                
                if obs_center[0] < 5.0 and abs(obs_center[1]) < 4.0:
                    
                    self.aviodance_state = True
                    self.avoid_right = obs_center[1] < 0
                    obs_direction = np.arctan2(obs_center[1],obs_center[0])
                    self.steering_mag = 0.7*( np.exp(- ( abs(obs_direction) - 1.4 ) ) - 1 )
                else:
                    self.aviodance_state = False
            else:
                self.get_logger().info("Identifier: pretty safe to cross")
                self.aviodance_state = False
            
        

    def lidar_callback(self, msg):
        self.go = True
        # Convert the PointCloud2 message to a list of XYZ points
        self.pc_data = pc2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)
        
        ## do the downsampling
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.pc_data)

        down_pcd = pcd.farthest_point_down_sample(8000)
        # down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        downsampled_points = np.asarray(down_pcd.points)

        self.pc_np = downsampled_points
        
        
        
    def error_state(self):
        x_current = self.x
        y_current = self.y
        theta_current = self.theta
        v_current = self.v
        
        #post process theta
        while theta_current<-np.pi:
            theta_current = theta_current+2*np.pi
        while theta_current>np.pi:
            theta_current = theta_current - 2*np.pi

        dist = np.zeros((1,len(self.ref_traj[:,1])))
        for i in range(len(self.ref_traj[:,1])):
            dist[0][i] = dist[0][i] = (x_current+np.cos(theta_current)*self.lookahead-self.ref_traj[i][0])**2+(y_current+np.sin(theta_current)*self.lookahead-self.ref_traj[i][1])**2
        index = dist.argmin()

        ref_state_current = list(self.ref_traj[index,:])
        err_theta = 0
        ref = ref_state_current[2]
        act = theta_current

        if( (ref>0 and act>0) or (ref<=0 and act <=0)):
            err_theta = ref-act
        elif( ref<=0 and act > 0):
            if(abs(ref-act)<abs(2*np.pi+ref-act)):
                err_theta = -abs(act-ref)
            else:
                err_theta = abs(2*np.pi + ref- act)
        else:
            if(abs(ref-act)<abs(2*np.pi-ref+act)):
                err_theta = abs(act-ref)
            else: 
                err_theta = -abs(2*np.pi-ref+act)


        RotM = np.array([ 
            [np.cos(-theta_current), -np.sin(-theta_current)],
            [np.sin(-theta_current), np.cos(-theta_current)]
        ])

        errM = np.array([[ref_state_current[0]-x_current],[ref_state_current[1]-y_current]])

        errRM = RotM@errM


        error_state = [errRM[0][0],errRM[1][0],err_theta, ref_state_current[3]-v_current]
        
        # suggested_steering = sum([x * y for x, y in zip(error_state, [0.02176878 , 0.72672704 , 0.78409284 ,-0.0105355 ])])
        # self.get_logger().info('suggested steering: %s' % suggested_steering)
        self.e_input = torch.tensor(np.array(error_state),dtype=torch.float32).unsqueeze(0)

        return error_state
    

    # callback to run a loop and publish data this class generates
    def pub_callback(self):
        if(not self.go):
            return
        ### for vehicle one
        msg = VehicleInput()
        ## get error state
        #e_flw = self.follow_error()
        e = self.error_state()
        
        #self.throttle = ctrl[0,0].item()
        
        ########### choose obstacle detection method:
        time_start = time.time()
        self.Obstacle_Detection_ptClustering()
        time_end = time.time()
        self.get_logger().info("lidar processing time: "+str(time_end-time_start))
        if self.v > 0.5:
            # # obstacle avoidance based on detection
            if self.aviodance_state:
                self.get_logger().info('avoiding !!!')
                if self.avoid_right:
                    steering = 0.45 * self.steering_mag
                    self.get_logger().info('Obstacle on your right, turn left')
                else:
                    steering = -0.45 * self.steering_mag
                    self.get_logger().info('Obstacle on your left, turn right')
                
            else:
                steering = sum([x * y for x, y in zip(e, [0.02176878 , 0.32672704 , 1.3 ,-0.0105355 ])])
                steering = np.clip(steering,-0.45,0.45)
        else:
            steering = 0.0
            throttle = 1.0
            msg.throttle = throttle
            self.pub_vehicle_cmd.publish(msg)
            self.get_logger().info('got stuck, getting up to speed')
            time.sleep(20)
        
        #steering = np.clip(steering,-0.8,0.8)
            # self.get_logger().info('safe')
        #steering = 0.0
        ##### ensure steering can't change too much between timesteps, smooth transition
        # delta_steering = steering - self.steering
        # if abs(delta_steering) > 0.175:
        #     self.steering = self.steering + 0.175 * delta_steering / abs(delta_steering)
        #     self.get_logger().info("!!!!!!!!!!!!!!!!!!steering changed too much, smoothing!!!!!!!!!!!!!!!!!!")
        # else:
        #     self.steering = steering
        self.steering = steering
        self.get_logger().info('steering:'+str(self.steering))
        
        
        
        
        msg.steering = np.clip(self.steering, -1.0, 1.0)
        msg.throttle = np.clip(self.throttle, 0, 1)
        msg.braking = np.clip(self.braking, 0, 1)
        #self.get_logger().info("sending vehicle inputs: %s" % msg)
        self.pub_vehicle_cmd.publish(msg)
        
        
        
        # # ## record data
        # # Save the point cloud to a PCD file
        # # pc_file_path = './pc/'+str(self.counter) + '.csv'
        ## np.savetxt(pc_file_path, self.pc_data, delimiter=',',fmt='%.8f')
        # ##self.get_logger().info("done saving data")
        # log_file_path = './trainingData/input_'+str(self.exp_index) + '.csv'
        # data_output = 'traj_output_.csv'
        # with open (log_file_path,'a', encoding='UTF8') as csvfile:
        #         my_writer = csv.writer(csvfile, quoting=csv.QUOTE_NONE, escapechar=' ')
        #         #for row in pt:
        #         my_writer.writerow([self.x, self.y, self.v,msg.throttle, self.bounding_box[0], self.bounding_box[1],self.bounding_box[2],self.engine_speed,self.engine_tq])
        #         csvfile.close()

        self.counter += 1

def main(args=None):
    rclpy.init(args=args)
    control = ControlNode()
    rclpy.spin(control)

    control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

