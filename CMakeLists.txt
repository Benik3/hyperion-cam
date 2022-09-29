cmake_minimum_required(VERSION 3.6)

project(Hyperion-cam
	DESCRIPTION "Camera grabber for Hyperion"
	LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "-Wall -Winvalid-pch -Wnon-virtual-dtor -Wextra -Werror -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(PkgConfig)

pkg_check_modules(LIBCAMERA REQUIRED IMPORTED_TARGET libcamera)
message(STATUS "libcamera library found:")
message(STATUS "    version: ${LIBCAMERA_VERSION}")
message(STATUS "    libraries: ${LIBCAMERA_LINK_LIBRARIES}")
message(STATUS "    include path: ${LIBCAMERA_INCLUDE_DIRS}")

pkg_check_modules(LIBEVENT REQUIRED IMPORTED_TARGET libevent_pthreads)
message(STATUS "libevent_pthreads library found:")
message(STATUS "    version: ${LIBEVENT_VERSION}")
message(STATUS "    libraries: ${LIBEVENT_LINK_LIBRARIES}")
message(STATUS "    include path: ${LIBEVENT_INCLUDE_DIRS}")

find_package(Threads REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED )
find_package(OpenCV REQUIRED)

add_library(FlatBuffer
    FlatBufferConnection.cpp
    FlatBufferConnection.h
)

add_library(opencvTransform
    opencvTransform.cpp
    opencvTransform.h
)

include_directories(${CMAKE_SOURCE_DIR} ${LIBCAMERA_INCLUDE_DIRS} ${LIBEVENT_INCLUDE_DIRS} ${Boost_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})

add_executable(hyperion-cam hyperion-cam.cpp event_loop.cpp)

target_link_libraries(hyperion-cam PkgConfig::LIBEVENT)
target_link_libraries(hyperion-cam PkgConfig::LIBCAMERA)
target_link_libraries(hyperion-cam ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(hyperion-cam ${Boost_LIBRARIES})
target_link_libraries(hyperion-cam ${OpenCV_LIBS})
target_link_libraries(hyperion-cam FlatBuffer)
target_link_libraries(hyperion-cam opencvTransform)