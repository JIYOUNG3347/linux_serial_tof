cmake_minimum_required(VERSION 2.4)
project(serial_port)

set(CMAKE_CXX_STANDARD 14)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(test test.cpp time_stamp.cpp time_stamp.h serial_port_stream.cpp serial_port_stream.h serial_port.cpp serial_port.h crc_calc.cpp crc_calc.h)

target_link_libraries(test pthread ${OpenCV_LIBS})
