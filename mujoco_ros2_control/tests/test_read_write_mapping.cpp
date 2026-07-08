// Copyright (C) 2026 ros2_control Development Team
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//         http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// These tests guard the joint<->actuator name-matching performed every read()/write() cycle
// (see MujocoSystemInterface::actuator_state_to_joint_state() and
// ::joint_command_to_actuator_command()). That matching is being changed from an O(joints x
// actuators) std::string comparison, redone every cycle, to a cached index computed once at
// on_init(). These tests characterize the existing (pre-refactor) behavior end-to-end through
// the public read()/write() API, so they must keep passing unchanged across that refactor.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <optional>
#include <thread>

#include <hardware_interface/version.h>
#include <mujoco/mujoco.h>
#include <hardware_interface/hardware_info.hpp>
#include <mujoco_ros2_control/mujoco_system_interface.hpp>
#include <mujoco_ros2_control_msgs/srv/set_pause.hpp>
#include <rclcpp/rclcpp.hpp>

#define ROS_DISTRO_HUMBLE (HARDWARE_INTERFACE_VERSION_MAJOR < 3)

namespace
{

// A single hinge joint driven by a MuJoCo position actuator named differently from the joint,
// so the test also exercises the actuator-name (not just joint-name) matching path.
constexpr const char* kTestModel = R"(<?xml version="1.0"?>
<mujoco model="test_read_write_mapping">
  <option timestep="0.002"/>

  <worldbody>
    <body name="pendulum" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.3 0 0" mass="1"/>
    </body>
  </worldbody>

  <actuator>
    <position name="hinge_pos" joint="hinge" kp="10"/>
  </actuator>
</mujoco>
)";

}  // namespace

class ReadWriteMappingTest : public ::testing::Test
{
protected:
  static void SetUpTestSuite()
  {
    if (!rclcpp::ok())
    {
      rclcpp::init(0, nullptr);
    }
  }

  static void TearDownTestSuite()
  {
    if (rclcpp::ok())
    {
      rclcpp::shutdown();
    }
  }

  void SetUp() override
  {
    test_model_path_ = "/tmp/test_read_write_mapping_model.xml";
    std::ofstream file(test_model_path_);
    file << kTestModel;
    file.close();

    interface_ = std::make_shared<mujoco_ros2_control::MujocoSystemInterface>();
  }

  void TearDown() override
  {
    if (interface_)
    {
      rclcpp_lifecycle::State inactive_state(0, "inactive");
      interface_->on_deactivate(inactive_state);
      interface_.reset();
    }
    if (std::filesystem::exists(test_model_path_))
    {
      std::filesystem::remove(test_model_path_);
    }
  }

  hardware_interface::HardwareInfo create_hardware_info()
  {
    hardware_interface::HardwareInfo info;
    info.name = "test_mujoco";
    info.type = "system";
    info.hardware_parameters["mujoco_model"] = test_model_path_;
    info.hardware_parameters["meshdir"] = "";
    info.hardware_parameters["headless"] = "true";
    info.hardware_parameters["disable_rendering"] = "true";

    // Joint "hinge" maps (via get_joint_actuator_name/get_actuator_id) directly to MuJoCo
    // actuator "hinge_pos", exercising the same name-matching path used by
    // actuator_state_to_joint_state()/joint_command_to_actuator_command().
    const auto make_interface = [](const std::string& name) {
      hardware_interface::InterfaceInfo interface_info;
      interface_info.name = name;
      return interface_info;
    };

    hardware_interface::ComponentInfo joint;
    joint.name = "hinge";
    joint.type = "joint";
    joint.command_interfaces.push_back(make_interface(hardware_interface::HW_IF_POSITION));
    joint.state_interfaces.push_back(make_interface(hardware_interface::HW_IF_POSITION));
    joint.state_interfaces.push_back(make_interface(hardware_interface::HW_IF_VELOCITY));
    joint.state_interfaces.push_back(make_interface(hardware_interface::HW_IF_EFFORT));
    info.joints.push_back(joint);

    return info;
  }

  bool init_and_wait_for_model()
  {
    hardware_info_ = create_hardware_info();
#if ROS_DISTRO_HUMBLE
    auto result = interface_->on_init(hardware_info_);
#else
    hardware_interface::HardwareComponentInterfaceParams params;
    params.hardware_info = hardware_info_;
    auto result = interface_->on_init(params);
#endif
    if (result != hardware_interface::CallbackReturn::SUCCESS)
    {
      return false;
    }

    const auto start = std::chrono::steady_clock::now();
    mjModel* model = nullptr;
    mjData* data = nullptr;
    while (std::chrono::steady_clock::now() - start < std::chrono::seconds(2))
    {
      interface_->get_model(model);
      interface_->get_data(data);
      if (model != nullptr && data != nullptr)
      {
        return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return false;
  }

  // Pauses the simulation via the internal node's `~/set_pause` service so that qpos/qvel are
  // stable while we assert on them. The internal node is always named
  // "mujoco_ros2_control_node" with no namespace (see MujocoSystemInterface::on_init()).
  bool pause_simulation()
  {
    auto client_node = std::make_shared<rclcpp::Node>("test_read_write_mapping_client");
    auto client = client_node->create_client<mujoco_ros2_control_msgs::srv::SetPause>(
        "/mujoco_ros2_control_node/set_pause");
    if (!client->wait_for_service(std::chrono::seconds(5)))
    {
      return false;
    }
    auto request = std::make_shared<mujoco_ros2_control_msgs::srv::SetPause::Request>();
    request->paused = true;
    auto future = client->async_send_request(request);
    if (rclcpp::spin_until_future_complete(client_node, future, std::chrono::seconds(5)) !=
        rclcpp::FutureReturnCode::SUCCESS)
    {
      return false;
    }
    // Give the physics loop (1 ms poll granularity) time to observe the pause and settle.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return future.get()->success;
  }

  std::optional<double> find_value(std::vector<hardware_interface::StateInterface>& interfaces,
                                   const std::string& interface_name)
  {
    for (auto& interface : interfaces)
    {
      if (interface.get_name() == interface_name)
      {
        return interface.get_optional<double>();
      }
    }
    return std::nullopt;
  }

  std::string test_model_path_;
  hardware_interface::HardwareInfo hardware_info_;
  std::shared_ptr<mujoco_ros2_control::MujocoSystemInterface> interface_;
};

// Guards joint_command_to_actuator_command(): a position command written to the joint's
// exported command interface must reach the matching MuJoCo actuator's `ctrl` entry. `ctrl` is
// only written by write(), never mutated by mj_step(), so this is deterministic even with the
// physics thread running in the background.
TEST_F(ReadWriteMappingTest, WriteMapsJointCommandToActuatorCtrl)
{
  ASSERT_TRUE(init_and_wait_for_model());

  auto command_interfaces = interface_->export_command_interfaces();
  auto position_command = std::find_if(
      command_interfaces.begin(), command_interfaces.end(),
      [](auto& interface) { return interface.get_name() == "hinge/" + std::string(hardware_interface::HW_IF_POSITION); });
  ASSERT_NE(position_command, command_interfaces.end());

  constexpr double kCommandedPosition = 0.42;
  ASSERT_TRUE(position_command->set_value<double>(kCommandedPosition));

  const rclcpp::Time time(0, 0, RCL_ROS_TIME);
  ASSERT_EQ(interface_->write(time, rclcpp::Duration(0, 0)), hardware_interface::return_type::OK);

  mjData* data = nullptr;
  interface_->get_data(data);
  ASSERT_NE(data, nullptr);
  EXPECT_DOUBLE_EQ(data->ctrl[0], kCommandedPosition);
  mj_deleteData(data);
}

// Guards actuator_state_to_joint_state(): the joint's exported position/velocity/effort state
// interfaces must reflect the underlying MuJoCo actuator's qpos/qvel/qfrc_actuator after read().
// The sim is paused and the physics state written directly so the expected values are known
// exactly.
TEST_F(ReadWriteMappingTest, ReadMapsActuatorStateToJointState)
{
  ASSERT_TRUE(init_and_wait_for_model());
  ASSERT_TRUE(pause_simulation());

  mjData* data = nullptr;
  interface_->get_data(data);
  ASSERT_NE(data, nullptr);

  constexpr double kQpos = 0.91;
  constexpr double kQvel = -0.35;
  constexpr double kQfrcActuator = 2.7;
  data->qpos[0] = kQpos;
  data->qvel[0] = kQvel;
  data->qfrc_actuator[0] = kQfrcActuator;
  interface_->set_data(data);
  mj_deleteData(data);

  const rclcpp::Time time(0, 0, RCL_ROS_TIME);
  ASSERT_EQ(interface_->read(time, rclcpp::Duration(0, 0)), hardware_interface::return_type::OK);

  auto state_interfaces = interface_->export_state_interfaces();
  const auto position = find_value(state_interfaces, "hinge/" + std::string(hardware_interface::HW_IF_POSITION));
  const auto velocity = find_value(state_interfaces, "hinge/" + std::string(hardware_interface::HW_IF_VELOCITY));
  const auto effort = find_value(state_interfaces, "hinge/" + std::string(hardware_interface::HW_IF_EFFORT));

  ASSERT_TRUE(position.has_value());
  ASSERT_TRUE(velocity.has_value());
  ASSERT_TRUE(effort.has_value());
  EXPECT_DOUBLE_EQ(position.value(), kQpos);
  EXPECT_DOUBLE_EQ(velocity.value(), kQvel);
  EXPECT_DOUBLE_EQ(effort.value(), kQfrcActuator);
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  int result = RUN_ALL_TESTS();
  return result;
}
