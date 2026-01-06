cmake_minimum_required(VERSION 3.10)

project(Hyperion-cam
	DESCRIPTION "Camera grabber for Hyperion"
	LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "-Wall -Winvalid-pch -Wnon-virtual-dtor -Wextra -Werror -Wno-unused-parameter")
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
        gpuNv12Transform.cpp
        gpuNv12Transform.h
        gpuRgbTransform.cpp
        gpuRgbTransform.h
    )
    target_link_libraries(gpuNv12Transform PkgConfig::EGL PkgConfig::GLESv2)
    target_compile_definitions(gpuNv12Transform PRIVATE HAVE_GPU_TRANSFORM=1)
else()
    message(STATUS "GPU transform: DISABLED")
endif()

add_library(FlatBuffer
    FlatBufferConnection.cpp
    FlatBufferConnection.h
)

add_library(opencvTransform
    opencvTransform.cpp
    opencvTransform.h
)

include_directories(${CMAKE_SOURCE_DIR} ${LIBCAMERA_INCLUDE_DIRS} ${LIBEVENT_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

add_executable(hyperion-cam hyperion-cam.cpp event_loop.cpp)

target_link_libraries(hyperion-cam PkgConfig::LIBEVENT)
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