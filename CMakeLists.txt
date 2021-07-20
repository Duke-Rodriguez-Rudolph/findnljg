CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT(demo2)

find_package(OpenCV REQUIRED)

ADD_EXECUTABLE(demo2  main.cpp)

target_link_libraries(demo2 PRIVATE ${OpenCV_LIBS})
