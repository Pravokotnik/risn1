cmake_minimum_required(VERSION 3.5)
project(dis_tutorial3)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)

find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)

find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Task.msg"
  "msg/FaceMsg.msg"
  DEPENDENCIES geometry_msgs
)


install(DIRECTORY
  launch
  meshes
  urdf
  config
  worlds
  maps
  DESTINATION share/${PROJECT_NAME}/
)

install(PROGRAMS
  scripts/arm_mover_actions.py
  scripts/detect_people.py
  scripts/detect_people_no_pc.py
  scripts/detect_people_no_pc_deepface.py
  scripts/robot_commander.py
  scripts/robot_commander_custom.py
  scripts/dp2.py
  scripts/clicked_point_listener.py
  scripts/start_round.py
  scripts/get_coords.py
  scripts/detect_rings.py
  scripts/detect_people_custom.py
  scripts/face_filter.py
  scripts/face_filter1.py
  scripts/face_filter2.py
  scripts/detect_rings_mod.py
  scripts/detect_rings1.py
  scripts/main.py
  scripts/main1.py
  scripts/main_irl.py
  scripts/get_rings.py
  scripts/rings_filter.py
  DESTINATION lib/${PROJECT_NAME})

ament_package()
