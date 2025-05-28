// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dis_tutorial3:msg/Task.idl
// generated code does not contain a copyright notice

#ifndef DIS_TUTORIAL3__MSG__DETAIL__TASK__STRUCT_H_
#define DIS_TUTORIAL3__MSG__DETAIL__TASK__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'task_type'
// Member 'description'
#include "rosidl_runtime_c/string.h"
// Member 'target_pose'
#include "geometry_msgs/msg/detail/pose_stamped__struct.h"

/// Struct defined in msg/Task in the package dis_tutorial3.
/**
  * tasks_msgs/msg/Task.msg
 */
typedef struct dis_tutorial3__msg__Task
{
  uint8_t priority;
  uint32_t id;
  /// "waypoint", "ring", "face", etc.
  rosidl_runtime_c__String task_type;
  geometry_msgs__msg__PoseStamped target_pose;
  rosidl_runtime_c__String description;
} dis_tutorial3__msg__Task;

// Struct for a sequence of dis_tutorial3__msg__Task.
typedef struct dis_tutorial3__msg__Task__Sequence
{
  dis_tutorial3__msg__Task * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dis_tutorial3__msg__Task__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DIS_TUTORIAL3__MSG__DETAIL__TASK__STRUCT_H_
