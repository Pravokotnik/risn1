// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from dis_tutorial3:msg/Task.idl
// generated code does not contain a copyright notice

#ifndef DIS_TUTORIAL3__MSG__DETAIL__TASK__STRUCT_HPP_
#define DIS_TUTORIAL3__MSG__DETAIL__TASK__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'target_pose'
#include "geometry_msgs/msg/detail/pose_stamped__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__dis_tutorial3__msg__Task __attribute__((deprecated))
#else
# define DEPRECATED__dis_tutorial3__msg__Task __declspec(deprecated)
#endif

namespace dis_tutorial3
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct Task_
{
  using Type = Task_<ContainerAllocator>;

  explicit Task_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : target_pose(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->priority = 0;
      this->id = 0ul;
      this->task_type = "";
      this->description = "";
    }
  }

  explicit Task_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : task_type(_alloc),
    target_pose(_alloc, _init),
    description(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->priority = 0;
      this->id = 0ul;
      this->task_type = "";
      this->description = "";
    }
  }

  // field types and members
  using _priority_type =
    uint8_t;
  _priority_type priority;
  using _id_type =
    uint32_t;
  _id_type id;
  using _task_type_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _task_type_type task_type;
  using _target_pose_type =
    geometry_msgs::msg::PoseStamped_<ContainerAllocator>;
  _target_pose_type target_pose;
  using _description_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _description_type description;

  // setters for named parameter idiom
  Type & set__priority(
    const uint8_t & _arg)
  {
    this->priority = _arg;
    return *this;
  }
  Type & set__id(
    const uint32_t & _arg)
  {
    this->id = _arg;
    return *this;
  }
  Type & set__task_type(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->task_type = _arg;
    return *this;
  }
  Type & set__target_pose(
    const geometry_msgs::msg::PoseStamped_<ContainerAllocator> & _arg)
  {
    this->target_pose = _arg;
    return *this;
  }
  Type & set__description(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->description = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    dis_tutorial3::msg::Task_<ContainerAllocator> *;
  using ConstRawPtr =
    const dis_tutorial3::msg::Task_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dis_tutorial3::msg::Task_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dis_tutorial3::msg::Task_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dis_tutorial3::msg::Task_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dis_tutorial3::msg::Task_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dis_tutorial3::msg::Task_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dis_tutorial3::msg::Task_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dis_tutorial3::msg::Task_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dis_tutorial3::msg::Task_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dis_tutorial3__msg__Task
    std::shared_ptr<dis_tutorial3::msg::Task_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dis_tutorial3__msg__Task
    std::shared_ptr<dis_tutorial3::msg::Task_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Task_ & other) const
  {
    if (this->priority != other.priority) {
      return false;
    }
    if (this->id != other.id) {
      return false;
    }
    if (this->task_type != other.task_type) {
      return false;
    }
    if (this->target_pose != other.target_pose) {
      return false;
    }
    if (this->description != other.description) {
      return false;
    }
    return true;
  }
  bool operator!=(const Task_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Task_

// alias to use template instance with default allocator
using Task =
  dis_tutorial3::msg::Task_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace dis_tutorial3

#endif  // DIS_TUTORIAL3__MSG__DETAIL__TASK__STRUCT_HPP_
