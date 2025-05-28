// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from dis_tutorial3:msg/Task.idl
// generated code does not contain a copyright notice

#ifndef DIS_TUTORIAL3__MSG__DETAIL__TASK__TRAITS_HPP_
#define DIS_TUTORIAL3__MSG__DETAIL__TASK__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "dis_tutorial3/msg/detail/task__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'target_pose'
#include "geometry_msgs/msg/detail/pose_stamped__traits.hpp"

namespace dis_tutorial3
{

namespace msg
{

inline void to_flow_style_yaml(
  const Task & msg,
  std::ostream & out)
{
  out << "{";
  // member: priority
  {
    out << "priority: ";
    rosidl_generator_traits::value_to_yaml(msg.priority, out);
    out << ", ";
  }

  // member: id
  {
    out << "id: ";
    rosidl_generator_traits::value_to_yaml(msg.id, out);
    out << ", ";
  }

  // member: task_type
  {
    out << "task_type: ";
    rosidl_generator_traits::value_to_yaml(msg.task_type, out);
    out << ", ";
  }

  // member: target_pose
  {
    out << "target_pose: ";
    to_flow_style_yaml(msg.target_pose, out);
    out << ", ";
  }

  // member: description
  {
    out << "description: ";
    rosidl_generator_traits::value_to_yaml(msg.description, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const Task & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: priority
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "priority: ";
    rosidl_generator_traits::value_to_yaml(msg.priority, out);
    out << "\n";
  }

  // member: id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "id: ";
    rosidl_generator_traits::value_to_yaml(msg.id, out);
    out << "\n";
  }

  // member: task_type
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "task_type: ";
    rosidl_generator_traits::value_to_yaml(msg.task_type, out);
    out << "\n";
  }

  // member: target_pose
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "target_pose:\n";
    to_block_style_yaml(msg.target_pose, out, indentation + 2);
  }

  // member: description
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "description: ";
    rosidl_generator_traits::value_to_yaml(msg.description, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const Task & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace dis_tutorial3

namespace rosidl_generator_traits
{

[[deprecated("use dis_tutorial3::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const dis_tutorial3::msg::Task & msg,
  std::ostream & out, size_t indentation = 0)
{
  dis_tutorial3::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use dis_tutorial3::msg::to_yaml() instead")]]
inline std::string to_yaml(const dis_tutorial3::msg::Task & msg)
{
  return dis_tutorial3::msg::to_yaml(msg);
}

template<>
inline const char * data_type<dis_tutorial3::msg::Task>()
{
  return "dis_tutorial3::msg::Task";
}

template<>
inline const char * name<dis_tutorial3::msg::Task>()
{
  return "dis_tutorial3/msg/Task";
}

template<>
struct has_fixed_size<dis_tutorial3::msg::Task>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<dis_tutorial3::msg::Task>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<dis_tutorial3::msg::Task>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // DIS_TUTORIAL3__MSG__DETAIL__TASK__TRAITS_HPP_
