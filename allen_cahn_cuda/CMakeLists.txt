cmake_minimum_required(VERSION 3.7)

#made using this great guide: https://cliutils.gitlab.io/modern-cmake/chapters/basics.html

if(${CMAKE_VERSION} VERSION_LESS 3.27)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
else()
    cmake_policy(VERSION 3.27)
endif()

project(AllenCahnCuda VERSION 0.1
                  DESCRIPTION "Cuda implementation of phase field model for crystal growth"
                  LANGUAGES C)

add_executable(app main_platform_dev.c lib/platform.h lib/platform_linux.c)
target_link_libraries(app m) #link math

#Run using:
#cmake -S . -B build
#cmake --build build