/**
 * Copyright (c) 2025, United States Government, as represented by the
 * Administrator of the National Aeronautics and Space Administration.
 *
 * All rights reserved.
 *
 * This software is licensed under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with the
 * License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

#pragma once

#include <string>
#include <vector>

#include "GLFW/glfw3.h"
#include "mujoco/mujoco.h"
#include "rclcpp/rclcpp.hpp"

#include "rclcpp/node.hpp"
#include "rclcpp/publisher.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"

namespace mujoco_ros2_control
{

struct CameraData
{
  mjvCamera mjv_cam;
  mjrRect viewport;

  std::string name;
  std::string frame_name;
  uint32_t width;
  uint32_t height;

  std::vector<uint8_t> image_buffer;
  std::vector<float> depth_buffer;
  std::vector<float> depth_buffer_flipped;

  sensor_msgs::msg::Image image;
  sensor_msgs::msg::Image depth_image;
  sensor_msgs::msg::CameraInfo camera_info;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_image_pub;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_pub;
};

class MujocoCameras
{
public:
  explicit MujocoCameras(rclcpp::Node::SharedPtr& node);

  void init(mjModel* mujoco_model);
  void update(mjModel* mujoco_model, mjData* mujoco_data);
  void close();
  void set_model_update_mutex(std::shared_ptr<std::mutex> model_update_mutex)
  {
    model_update_mutex_ = model_update_mutex;
  };

private:
  void register_cameras(const mjModel* mujoco_model);

  rclcpp::Node::SharedPtr node_;

  // Rendering options for the cameras, currently hard coded to defaults
  mjvOption mjv_opt_;
  mjvScene mjv_scn_;
  mjrContext mjr_con_;

  // Containers for camera data and ROS constructs
  std::vector<CameraData> cameras_;

  std::shared_ptr<std::mutex> model_update_mutex_;
};

}  // namespace mujoco_ros2_control
