# tasks_msgs/msg/Task.msg
uint8 priority
uint32 id
string task_type  # "waypoint", "ring", "face", etc.
geometry_msgs/PoseStamped target_pose
string description