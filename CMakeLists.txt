cmake_minimum_required(VERSION 3.10)

project(Hyperion-cam
	DESCRIPTION "Camera grabber for Hyperion"
	LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

# -Wno-psabi silences GCC's informational ARM ABI note about the C++14→C++17
# change in how small aggregates (e.g. std::pair<double,double>) are passed/
# returned; the whole binary uses one compiler/std, so there is no real mismatch.
set(CMAKE_CXX_FLAGS "-Wall -Winvalid-pch -Wnon-virtual-dtor -Wextra -Werror -Wno-unused-parameter -Wno-psabi")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -fno-fast-math")
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

# evhttp (LED-calib TV web page) lives in libevent_extra, not _pthreads/_core.
pkg_check_modules(LIBEVENT_EXTRA REQUIRED IMPORTED_TARGET libevent_extra)

find_package(Threads REQUIRED)
find_package(Boost CONFIG REQUIRED COMPONENTS program_options)
find_package(OpenCV REQUIRED)

# Optional GPU accelerated NV12 transform (headless EGL + OpenGL ES 3)
pkg_check_modules(EGL QUIET IMPORTED_TARGET egl)
pkg_check_modules(GLESv2 QUIET IMPORTED_TARGET glesv2)

message(STATUS "GPU transform deps (pkg-config):")
if (EGL_FOUND)
    message(STATUS "  - egl: FOUND (version: ${EGL_VERSION})")
else()
    message(STATUS "  - egl: NOT FOUND (install libegl1-mesa-dev)")
endif()

if (GLESv2_FOUND)
    message(STATUS "  - glesv2: FOUND (version: ${GLESv2_VERSION})")
else()
    message(STATUS "  - glesv2: NOT FOUND (install libgles2-mesa-dev)")
endif()

set(HAVE_GPU_TRANSFORM 0)
if (EGL_FOUND AND GLESv2_FOUND)
    set(HAVE_GPU_TRANSFORM 1)
    message(STATUS "GPU transform: ENABLED")
    add_library(gpuNv12Transform
        src/gpuNv12Transform.cpp
        src/gpuNv12Transform.h
        src/gpuRgbTransform.cpp
        src/gpuRgbTransform.h
    )
    target_link_libraries(gpuNv12Transform PkgConfig::EGL PkgConfig::GLESv2)
    target_compile_definitions(gpuNv12Transform PRIVATE HAVE_GPU_TRANSFORM=1)
else()
    message(STATUS "GPU transform: DISABLED")
endif()

add_library(FlatBuffer
    src/FlatBufferConnection.cpp
    src/FlatBufferConnection.h
)

add_library(opencvTransform
    src/opencvTransform.cpp
    src/opencvTransform.h
)

include_directories(${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src ${LIBCAMERA_INCLUDE_DIRS} ${LIBEVENT_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

add_executable(hyperion-cam src/hyperion-cam.cpp src/event_loop.cpp src/ledCalib.cpp src/ledCalibFit.cpp src/ledCalibWeb.cpp src/cameraResponse.cpp)

target_link_libraries(hyperion-cam PkgConfig::LIBEVENT)
target_link_libraries(hyperion-cam PkgConfig::LIBEVENT_EXTRA)
target_link_libraries(hyperion-cam PkgConfig::LIBCAMERA)
target_link_libraries(hyperion-cam ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(hyperion-cam Boost::program_options)
target_link_libraries(hyperion-cam ${OpenCV_LIBS})
target_link_libraries(hyperion-cam FlatBuffer)
target_link_libraries(hyperion-cam opencvTransform)

if (HAVE_GPU_TRANSFORM)
    target_compile_definitions(hyperion-cam PRIVATE HAVE_GPU_TRANSFORM=1)
    target_link_libraries(hyperion-cam gpuNv12Transform)
endif()

# 32-bit ARM requires explicit libatomic for 64-bit std::atomic operations.
include(CheckCXXSourceCompiles)
check_cxx_source_compiles("
#include <atomic>
#include <cstdint>
std::atomic<int64_t> x{0};
int main() { return (int)x.load(); }
" HAVE_BUILTIN_ATOMIC64)
if (NOT HAVE_BUILTIN_ATOMIC64)
    target_link_libraries(hyperion-cam atomic)
endif()