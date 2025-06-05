#include "mujoco_ros2_simulation/mujoco_system_interface.hpp"
#include "array_safety.h"

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <chrono>
#include <future>
#include <stdexcept>

#include <rclcpp/rclcpp.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <hardware_interface/types/hardware_interface_type_values.hpp>

#define MUJOCO_PLUGIN_DIR "mujoco_plugin"

using namespace std::chrono_literals;

// constants
const double syncMisalign = 0.1;        // maximum mis-alignment before re-sync (simulation seconds)
const double simRefreshFraction = 0.7;  // fraction of refresh available for simulation
const int kErrorLength = 1024;          // load error string length

using Seconds = std::chrono::duration<double>;

namespace mujoco_ros2_simulation
{
namespace mj = ::mujoco;
namespace mju = ::mujoco::sample_util;

//---------------------------------------- plugin handling -----------------------------------------

// return the path to the directory containing the current executable
// used to determine the location of auto-loaded plugin libraries
std::string getExecutableDir() {
  constexpr char kPathSep = '/';
  const char* path = "/proc/self/exe";

  std::string realpath = [&]() -> std::string {
    std::unique_ptr<char[]> realpath(nullptr);
    std::uint32_t buf_size = 128;
    bool success = false;
    while (!success) {
      realpath.reset(new(std::nothrow) char[buf_size]);
      if (!realpath) {
        std::cerr << "cannot allocate memory to store executable path\n";
        return "";
      }

      std::size_t written = readlink(path, realpath.get(), buf_size);
      if (written < buf_size) {
        realpath.get()[written] = '\0';
        success = true;
      } else if (written == -1) {
        if (errno == EINVAL) {
          // path is already not a symlink, just use it
          return path;
        }

        std::cerr << "error while resolving executable path: " << strerror(errno) << '\n';
        return "";
      } else {
        // realpath is too small, grow and retry
        buf_size *= 2;
      }
    }
    return realpath.get();
  }();

  if (realpath.empty()) {
    return "";
  }

  for (std::size_t i = realpath.size() - 1; i > 0; --i) {
    if (realpath.c_str()[i] == kPathSep) {
      return realpath.substr(0, i);
    }
  }

  // don't scan through the entire file system's root
  return "";
}

// scan for libraries in the plugin directory to load additional plugins
void scanPluginLibraries() {
  // check and print plugins that are linked directly into the executable
  int nplugin = mjp_pluginCount();
  if (nplugin) {
    std::printf("Built-in plugins:\n");
    for (int i = 0; i < nplugin; ++i) {
      std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
    }
  }

  const std::string sep = "/";

  // try to open the ${EXECDIR}/MUJOCO_PLUGIN_DIR directory
  // ${EXECDIR} is the directory containing the simulate binary itself
  // MUJOCO_PLUGIN_DIR is the MUJOCO_PLUGIN_DIR preprocessor macro
  const std::string executable_dir = getExecutableDir();
  if (executable_dir.empty()) {
    return;
  }

  const std::string plugin_dir = getExecutableDir() + sep + MUJOCO_PLUGIN_DIR;
  mj_loadAllPluginLibraries(
      plugin_dir.c_str(), +[](const char* filename, int first, int count) {
        std::printf("Plugins registered by library '%s':\n", filename);
        for (int i = first; i < first + count; ++i) {
          std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
        }
      });
}


//------------------------------------------- simulation -------------------------------------------

const char* Diverged(int disableflags, const mjData* d) {
  if (disableflags & mjDSBL_AUTORESET) {
    for (mjtWarning w : {mjWARN_BADQACC, mjWARN_BADQVEL, mjWARN_BADQPOS}) {
      if (d->warning[w].number > 0) {
        return mju_warningText(w, d->warning[w].lastinfo);
      }
    }
  }
  return nullptr;
}

mjModel* LoadModel(const char* file, mj::Simulate& sim) {
  // this copy is needed so that the mju::strlen call below compiles
  char filename[mj::Simulate::kMaxFilenameLength];
  mju::strcpy_arr(filename, file);

  // make sure filename is not empty
  if (!filename[0]) {
    return nullptr;
  }

  // load and compile
  char loadError[kErrorLength] = "";
  mjModel* mnew = 0;
  auto load_start = mj::Simulate::Clock::now();
  if (mju::strlen_arr(filename)>4 &&
      !std::strncmp(filename + mju::strlen_arr(filename) - 4, ".mjb",
                    mju::sizeof_arr(filename) - mju::strlen_arr(filename)+4)) {
    mnew = mj_loadModel(filename, nullptr);
    if (!mnew) {
      mju::strcpy_arr(loadError, "could not load binary model");
    }
  } else {
    mnew = mj_loadXML(filename, nullptr, loadError, kErrorLength);

    // remove trailing newline character from loadError
    if (loadError[0]) {
      int error_length = mju::strlen_arr(loadError);
      if (loadError[error_length-1] == '\n') {
        loadError[error_length-1] = '\0';
      }
    }
  }
  auto load_interval = mj::Simulate::Clock::now() - load_start;
  double load_seconds = Seconds(load_interval).count();

  if (!mnew) {
    std::printf("%s\n", loadError);
    mju::strcpy_arr(sim.load_error, loadError);
    return nullptr;
  }

  // compiler warning: print and pause
  if (loadError[0]) {
    // mj_forward() below will print the warning message
    std::printf("Model compiled, but simulation warning (paused):\n  %s\n", loadError);
    sim.run = 0;
  }

  // if no error and load took more than 1/4 seconds, report load time
  else if (load_seconds > 0.25) {
    mju::sprintf_arr(loadError, "Model loaded in %.2g seconds", load_seconds);
  }

  mju::strcpy_arr(sim.load_error, loadError);

  return mnew;
}

MujocoSystemInterface::MujocoSystemInterface() = default;

MujocoSystemInterface::~MujocoSystemInterface()
{
  // If sim_ is created and running, clean shut it down
  if (sim_) {
    sim_->exitrequest.store(true);

    if (physics_thread_.joinable()) {
      physics_thread_.join();
    }
    if (ui_thread_.joinable()) {
      ui_thread_.join();
    }
  }

  // Cleanup data and the model, if they haven't been
  if (mj_data_) {
    mj_deleteData(mj_data_);
  }
  if (mj_model_) {
    mj_deleteModel(mj_model_);
  }
}

// simulate in background thread (while rendering in main thread)
void MujocoSystemInterface::PhysicsLoop(mj::Simulate& sim)
{
  // cpu-sim syncronization point
  std::chrono::time_point<mj::Simulate::Clock> syncCPU;
  mjtNum syncSim = 0;

  // run until asked to exit
  while (!sim_->exitrequest.load()) {
    if (sim_->droploadrequest.load()) {
      sim_->LoadMessage(sim_->dropfilename);
      mjModel* mnew = LoadModel(sim_->dropfilename, *sim_);
      sim_->droploadrequest.store(false);

      mjData* dnew = nullptr;
      if (mnew) dnew = mj_makeData(mnew);
      if (dnew) {
        sim_->Load(mnew, dnew, sim_->dropfilename);

        // lock the sim mutex
        const std::unique_lock<std::recursive_mutex> lock(*sim_mutex_);

        mj_deleteData(mj_data_);
        mj_deleteModel(mj_model_);

        mj_model_ = mnew;
        mj_data_ = dnew;
        mj_forward(mj_model_, mj_data_);

      } else {
        sim_->LoadMessageClear();
      }
    }

    if (sim_->uiloadrequest.load()) {
      sim_->uiloadrequest.fetch_sub(1);
      sim_->LoadMessage(sim_->filename);
      mjModel* mnew = LoadModel(sim_->filename, sim);
      mjData* dnew = nullptr;
      if (mnew) dnew = mj_makeData(mnew);
      if (dnew) {
        sim_->Load(mnew, dnew, sim_->filename);

        // lock the sim mutex
        const std::unique_lock<std::recursive_mutex> lock(*sim_mutex_);

        mj_deleteData(mj_data_);
        mj_deleteModel(mj_model_);

        mj_model_ = mnew;
        mj_data_ = dnew;
        mj_forward(mj_model_, mj_data_);

      } else {
        sim_->LoadMessageClear();
      }
    }

    // sleep for 1 ms or yield, to let main thread run
    //  yield results in busy wait - which has better timing but kills battery life
    if (sim_->run && sim_->busywait) {
      std::this_thread::yield();
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    {
      // lock the sim mutex
      const std::unique_lock<std::recursive_mutex> lock(sim_->mtx);

      // run only if model is present
      if (mj_model_) {
        // running
        if (sim_->run) {
          bool stepped = false;

          // record cpu time at start of iteration
          const auto startCPU = mj::Simulate::Clock::now();

          // elapsed CPU and simulation time since last sync
          const auto elapsedCPU = startCPU - syncCPU;
          double elapsedSim = mj_data_->time - syncSim;

          // requested slow-down factor
          double slowdown = 100 / sim_->percentRealTime[sim_->real_time_index];

          // misalignment condition: distance from target sim time is bigger than syncmisalign
          bool misaligned =
              std::abs(Seconds(elapsedCPU).count()/slowdown - elapsedSim) > syncMisalign;

          // out-of-sync (for any reason): reset sync times, step
          if (elapsedSim < 0 || elapsedCPU.count() < 0 || syncCPU.time_since_epoch().count() == 0 ||
              misaligned || sim_->speed_changed) {
            // re-sync
            syncCPU = startCPU;
            syncSim = mj_data_->time;
            sim_->speed_changed = false;

            // run single step, let next iteration deal with timing
            mj_step(mj_model_, mj_data_);
            const char* message = Diverged(mj_model_->opt.disableflags, mj_data_);
            if (message) {
              sim_->run = 0;
              mju::strcpy_arr(sim_->load_error, message);
            } else {
              stepped = true;
            }
          }

          // in-sync: step until ahead of cpu
          else {
            bool measured = false;
            mjtNum prevSim = mj_data_->time;

            double refreshTime = simRefreshFraction/sim_->refresh_rate;

            // step while sim lags behind cpu and within refreshTime
            while (Seconds((mj_data_->time - syncSim)*slowdown) < mj::Simulate::Clock::now() - syncCPU &&
                   mj::Simulate::Clock::now() - startCPU < Seconds(refreshTime)) {
              // measure slowdown before first step
              if (!measured && elapsedSim) {
                sim_->measured_slowdown =
                    std::chrono::duration<double>(elapsedCPU).count() / elapsedSim;
                measured = true;
              }

              // inject noise
              sim_->InjectNoise();

              // call mj_step
              mj_step(mj_model_, mj_data_);
              const char* message = Diverged(mj_model_->opt.disableflags, mj_data_);
              if (message) {
                sim_->run = 0;
                mju::strcpy_arr(sim_->load_error, message);
              } else {
                stepped = true;
              }

              // break if reset
              if (mj_data_->time < prevSim) {
                break;
              }
            }
          }

          // save current state to history buffer
          if (stepped) {
            sim_->AddToHistory();
          }
        }

        // paused
        else {
          // run mj_forward, to update rendering and joint sliders
          mj_forward(mj_model_, mj_data_);
          sim_->speed_changed = true;
        }
      }
    }  // release std::lock_guard<std::mutex>
  }
}

hardware_interface::CallbackReturn MujocoSystemInterface::on_init(
  const hardware_interface::HardwareInfo & info)
{
  if (hardware_interface::SystemInterface::on_init(info) != hardware_interface::CallbackReturn::SUCCESS) {
    return hardware_interface::CallbackReturn::ERROR;
  }
  system_info_ = info;

  // Load the model path from hardware parameters
  if (info.hardware_parameters.count("mujoco_model") == 0) {
    RCLCPP_FATAL(rclcpp::get_logger("MujocoSystemInterface"), "Missing 'mujoco_model' in <hardware_parameters>.");
    return hardware_interface::CallbackReturn::ERROR;
  }
  model_path_ = info.hardware_parameters.at("mujoco_model");

  RCLCPP_INFO_STREAM(rclcpp::get_logger("MujocoSystemInterface"), "Loading 'mujoco_model' from: " << model_path_);

  // We essentially reconstruct the 'simulate.cc::main()' function here, and launch a Simulate
  // object with all necessary rendering process/options attached.

  // scan for libraries in the plugin directory to load additional plugins
  scanPluginLibraries();

  // Retain scope
  mjv_defaultCamera(&cam_);
  mjv_defaultOption(&opt_);
  mjv_defaultPerturb(&pert_);

  // There is a timing issue here as the rendering context must be attached to the executing thread,
  // but we require the simulation to be available on init. So we spawn the sim in the rendering thread
  // prior to proceeding with initilization.
  auto sim_ready = std::make_shared<std::promise<void>>();
  std::future<void> sim_ready_future = sim_ready->get_future();

  // Launch the UI loop in the background
  ui_thread_ = std::thread([this, sim_ready]()
  {
    sim_ = std::make_unique<mj::Simulate>(
      std::make_unique<mj::GlfwAdapter>(),
      &cam_, &opt_, &pert_, /* is_passive = */ false
    );
    // Notify sim that we are ready
    sim_ready->set_value();

    RCLCPP_INFO(rclcpp::get_logger("MujocoSystemInterface"), "Starting the mujoco rendering thread...");
    // Blocks until terminated
    sim_->RenderLoop();
  });

  if (sim_ready_future.wait_for(2s) == std::future_status::timeout)
  {
    RCLCPP_FATAL(rclcpp::get_logger("MujocoSystemInterface"),
      "Timed out waiting to start simulation rendering!");
    return hardware_interface::CallbackReturn::ERROR;
  }

  // We maintain a pointer to the mutex so that we can lock from here, too.
  // Is this a terrible idea? Maybe, but it lets us use their libraries as is...
  sim_mutex_ = &sim_->mtx;

  // From here, we wrap up the PhysicsThread's starting function before beginning the loop in activate
  sim_->LoadMessage(model_path_.c_str());
  mj_model_ = LoadModel(model_path_.c_str(), *sim_);
  if (!mj_model_) {
    RCLCPP_FATAL(rclcpp::get_logger("MujocoSystemInterface"), "Mujoco failed to load '%s'", model_path_.c_str());
    return hardware_interface::CallbackReturn::ERROR;
  }

  {
    std::unique_lock<std::recursive_mutex> lock(*sim_mutex_);
    mj_data_ = mj_makeData(mj_model_);
  }
  if (!mj_data_) {
    RCLCPP_FATAL(rclcpp::get_logger("MujocoSystemInterface"), "Could not allocate mjData for '%s'", model_path_.c_str());
    return hardware_interface::CallbackReturn::ERROR;
  }

  // Get joint information and map it to our pointers
  n_joints_ = system_info_.joints.size();
  joint_names_.resize(n_joints_);
  muj_joint_id_.resize(n_joints_);
  muj_actuator_id_.resize(n_joints_);
  hw_positions_.assign(n_joints_, 0.0);
  hw_velocities_.assign(n_joints_, 0.0);
  hw_efforts_.assign(n_joints_, 0.0);
  hw_commands_.assign(n_joints_, 0.0);

  // Configure joint information
  for (size_t i = 0; i < n_joints_; ++i)
  {
    joint_names_[i] = system_info_.joints[i].name;

    // Precompute joint_id and actuator_id for faster read/write:
    muj_joint_id_[i] = mj_name2id(mj_model_, mjOBJ_JOINT, joint_names_[i].c_str());
    if (muj_joint_id_[i] < 0) {
      RCLCPP_ERROR(rclcpp::get_logger("MujocoSystemInterface"),
        "Joint '%s' not found in MuJoCo model", joint_names_[i].c_str());
      return hardware_interface::CallbackReturn::ERROR;
    }

    muj_actuator_id_[i] = mj_name2id(mj_model_, mjOBJ_ACTUATOR, joint_names_[i].c_str());
    if (muj_actuator_id_[i] < 0) {
      RCLCPP_ERROR(rclcpp::get_logger("MujocoSystemInterface"),
        "Actuator '%s' not found in MuJoCo model", joint_names_[i].c_str());
      return hardware_interface::CallbackReturn::ERROR;
    }
  }

  // When the interface is activated, we start the physics engine.
  physics_thread_ = std::thread([this]()
  {
    // Load the simulation and do an initial forward pass
    RCLCPP_INFO(rclcpp::get_logger("MujocoSystemInterface"), "Starting the mujoco physics thread...");
    sim_->Load(mj_model_, mj_data_, model_path_.c_str());
    // lock the sim mutex
    {
      const std::unique_lock<std::recursive_mutex> lock(*sim_mutex_);
      mj_forward(mj_model_, mj_data_);
    }
    // Blocks until terminated
    PhysicsLoop(*sim_);
  });

  return hardware_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface>
MujocoSystemInterface::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> state_interfaces;
  state_interfaces.reserve(n_joints_ * 3);

  for (size_t i = 0; i < n_joints_; ++i)
  {
    state_interfaces.emplace_back(joint_names_[i], hardware_interface::HW_IF_POSITION, &hw_positions_[i]);
    state_interfaces.emplace_back(joint_names_[i], hardware_interface::HW_IF_VELOCITY, &hw_velocities_[i]);
    state_interfaces.emplace_back(joint_names_[i], hardware_interface::HW_IF_EFFORT, &hw_efforts_[i]);
  }
  return state_interfaces;
}

std::vector<hardware_interface::CommandInterface>
MujocoSystemInterface::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> command_interfaces;
  command_interfaces.reserve(n_joints_);

  // TODO: Other command types?
  for (size_t i = 0; i < n_joints_; ++i)
  {
    command_interfaces.emplace_back(joint_names_[i], hardware_interface::HW_IF_POSITION, &hw_commands_[i]);
  }
  return command_interfaces;
}

hardware_interface::CallbackReturn MujocoSystemInterface::on_activate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  RCLCPP_INFO(rclcpp::get_logger("MujocoSystemInterface"),
              "Activating MuJoCo hardware interface and starting Simulate threads...");

  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn MujocoSystemInterface::on_deactivate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  RCLCPP_INFO(rclcpp::get_logger("MujocoSystemInterface"),
              "Deactivating MuJoCo hardware interface and shutting down Simulate...");

  // TODO: Should we shut things down here or in the destructor?

  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::return_type MujocoSystemInterface::read(
  const rclcpp::Time & /*time*/,
  const rclcpp::Duration & /*period*/)
{
  std::lock_guard<std::recursive_mutex> lock(*sim_mutex_);

  // Pull joint data from actuators
  for (size_t i = 0; i < n_joints_; ++i) {
    int j_id = muj_joint_id_[i];

    int qposadr = mj_model_->jnt_qposadr[j_id];
    hw_positions_[i] = mj_data_->qpos[qposadr];

    int qveladr = mj_model_->jnt_dofadr[j_id];
    hw_velocities_[i] = mj_data_->qvel[qveladr];

    hw_efforts_[i] = mj_data_->actuator_force[muj_actuator_id_[i]];
  }
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type MujocoSystemInterface::write(
  const rclcpp::Time & /*time*/,
  const rclcpp::Duration & /*period*/)
{
  std::lock_guard<std::recursive_mutex> lock(*sim_mutex_);

  // Set commads for actuators
  for (size_t i = 0; i < n_joints_; ++i) {
    int a_id = muj_actuator_id_[i];
    mj_data_->ctrl[a_id] = hw_commands_[i];
  }

  return hardware_interface::return_type::OK;
}

} // end namespace

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(
  mujoco_ros2_simulation::MujocoSystemInterface,
  hardware_interface::SystemInterface
);
