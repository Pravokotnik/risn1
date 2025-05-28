// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from dis_tutorial3:msg/Task.idl
// generated code does not contain a copyright notice
#include "dis_tutorial3/msg/detail/task__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `task_type`
// Member `description`
#include "rosidl_runtime_c/string_functions.h"
// Member `target_pose`
#include "geometry_msgs/msg/detail/pose_stamped__functions.h"

bool
dis_tutorial3__msg__Task__init(dis_tutorial3__msg__Task * msg)
{
  if (!msg) {
    return false;
  }
  // priority
  // id
  // task_type
  if (!rosidl_runtime_c__String__init(&msg->task_type)) {
    dis_tutorial3__msg__Task__fini(msg);
    return false;
  }
  // target_pose
  if (!geometry_msgs__msg__PoseStamped__init(&msg->target_pose)) {
    dis_tutorial3__msg__Task__fini(msg);
    return false;
  }
  // description
  if (!rosidl_runtime_c__String__init(&msg->description)) {
    dis_tutorial3__msg__Task__fini(msg);
    return false;
  }
  return true;
}

void
dis_tutorial3__msg__Task__fini(dis_tutorial3__msg__Task * msg)
{
  if (!msg) {
    return;
  }
  // priority
  // id
  // task_type
  rosidl_runtime_c__String__fini(&msg->task_type);
  // target_pose
  geometry_msgs__msg__PoseStamped__fini(&msg->target_pose);
  // description
  rosidl_runtime_c__String__fini(&msg->description);
}

bool
dis_tutorial3__msg__Task__are_equal(const dis_tutorial3__msg__Task * lhs, const dis_tutorial3__msg__Task * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // priority
  if (lhs->priority != rhs->priority) {
    return false;
  }
  // id
  if (lhs->id != rhs->id) {
    return false;
  }
  // task_type
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->task_type), &(rhs->task_type)))
  {
    return false;
  }
  // target_pose
  if (!geometry_msgs__msg__PoseStamped__are_equal(
      &(lhs->target_pose), &(rhs->target_pose)))
  {
    return false;
  }
  // description
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->description), &(rhs->description)))
  {
    return false;
  }
  return true;
}

bool
dis_tutorial3__msg__Task__copy(
  const dis_tutorial3__msg__Task * input,
  dis_tutorial3__msg__Task * output)
{
  if (!input || !output) {
    return false;
  }
  // priority
  output->priority = input->priority;
  // id
  output->id = input->id;
  // task_type
  if (!rosidl_runtime_c__String__copy(
      &(input->task_type), &(output->task_type)))
  {
    return false;
  }
  // target_pose
  if (!geometry_msgs__msg__PoseStamped__copy(
      &(input->target_pose), &(output->target_pose)))
  {
    return false;
  }
  // description
  if (!rosidl_runtime_c__String__copy(
      &(input->description), &(output->description)))
  {
    return false;
  }
  return true;
}

dis_tutorial3__msg__Task *
dis_tutorial3__msg__Task__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dis_tutorial3__msg__Task * msg = (dis_tutorial3__msg__Task *)allocator.allocate(sizeof(dis_tutorial3__msg__Task), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(dis_tutorial3__msg__Task));
  bool success = dis_tutorial3__msg__Task__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
dis_tutorial3__msg__Task__destroy(dis_tutorial3__msg__Task * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    dis_tutorial3__msg__Task__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
dis_tutorial3__msg__Task__Sequence__init(dis_tutorial3__msg__Task__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dis_tutorial3__msg__Task * data = NULL;

  if (size) {
    data = (dis_tutorial3__msg__Task *)allocator.zero_allocate(size, sizeof(dis_tutorial3__msg__Task), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = dis_tutorial3__msg__Task__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        dis_tutorial3__msg__Task__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
dis_tutorial3__msg__Task__Sequence__fini(dis_tutorial3__msg__Task__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      dis_tutorial3__msg__Task__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

dis_tutorial3__msg__Task__Sequence *
dis_tutorial3__msg__Task__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dis_tutorial3__msg__Task__Sequence * array = (dis_tutorial3__msg__Task__Sequence *)allocator.allocate(sizeof(dis_tutorial3__msg__Task__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = dis_tutorial3__msg__Task__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
dis_tutorial3__msg__Task__Sequence__destroy(dis_tutorial3__msg__Task__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    dis_tutorial3__msg__Task__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
dis_tutorial3__msg__Task__Sequence__are_equal(const dis_tutorial3__msg__Task__Sequence * lhs, const dis_tutorial3__msg__Task__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!dis_tutorial3__msg__Task__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
dis_tutorial3__msg__Task__Sequence__copy(
  const dis_tutorial3__msg__Task__Sequence * input,
  dis_tutorial3__msg__Task__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(dis_tutorial3__msg__Task);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    dis_tutorial3__msg__Task * data =
      (dis_tutorial3__msg__Task *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!dis_tutorial3__msg__Task__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          dis_tutorial3__msg__Task__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!dis_tutorial3__msg__Task__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
