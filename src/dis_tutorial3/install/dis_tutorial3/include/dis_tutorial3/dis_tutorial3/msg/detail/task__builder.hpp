// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dis_tutorial3:msg/Task.idl
// generated code does not contain a copyright notice

#ifndef DIS_TUTORIAL3__MSG__DETAIL__TASK__BUILDER_HPP_
#define DIS_TUTORIAL3__MSG__DETAIL__TASK__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dis_tutorial3/msg/detail/task__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dis_tutorial3
{

namespace msg
{

namespace builder
{

class Init_Task_description
{
public:
  explicit Init_Task_description(::dis_tutorial3::msg::Task & msg)
  : msg_(msg)
  {}
  ::dis_tutorial3::msg::Task description(::dis_tutorial3::msg::Task::_description_type arg)
  {
    msg_.description = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dis_tutorial3::msg::Task msg_;
};

class Init_Task_target_pose
{
public:
  explicit Init_Task_target_pose(::dis_tutorial3::msg::Task & msg)
  : msg_(msg)
  {}
  Init_Task_description target_pose(::dis_tutorial3::msg::Task::_target_pose_type arg)
  {
    msg_.target_pose = std::move(arg);
    return Init_Task_description(msg_);
  }

private:
  ::dis_tutorial3::msg::Task msg_;
};

class Init_Task_task_type
{
public:
  explicit Init_Task_task_type(::dis_tutorial3::msg::Task & msg)
  : msg_(msg)
  {}
  Init_Task_target_pose task_type(::dis_tutorial3::msg::Task::_task_type_type arg)
  {
    msg_.task_type = std::move(arg);
    return Init_Task_target_pose(msg_);
  }

private:
  ::dis_tutorial3::msg::Task msg_;
};

class Init_Task_id
{
public:
  explicit Init_Task_id(::dis_tutorial3::msg::Task & msg)
  : msg_(msg)
  {}
  Init_Task_task_type id(::dis_tutorial3::msg::Task::_id_type arg)
  {
    msg_.id = std::move(arg);
    return Init_Task_task_type(msg_);
  }

private:
  ::dis_tutorial3::msg::Task msg_;
};

class Init_Task_priority
{
public:
  Init_Task_priority()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Task_id priority(::dis_tutorial3::msg::Task::_priority_type arg)
  {
    msg_.priority = std::move(arg);
    return Init_Task_id(msg_);
  }

private:
  ::dis_tutorial3::msg::Task msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::dis_tutorial3::msg::Task>()
{
  return dis_tutorial3::msg::builder::Init_Task_priority();
}

}  // namespace dis_tutorial3

#endif  // DIS_TUTORIAL3__MSG__DETAIL__TASK__BUILDER_HPP_
