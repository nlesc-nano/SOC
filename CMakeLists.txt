cmake_minimum_required(VERSION 3.16)
project(SOC LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_definitions(EIGEN_MPL2_ONLY EIGEN_NO_DEBUG)

# Find dependencies
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3   CONFIG REQUIRED)
find_package(libint2  CONFIG REQUIRED)

# Ensure the extension is placed in src/
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/src/soc)

pybind11_add_module(libint_cpp
    libint/bindings.cpp
    libint/integrals_core.cpp
)

target_link_libraries(libint_cpp PRIVATE
    Libint2::cxx
    Eigen3::Eigen
)

get_target_property(LIBINT2_LOC Libint2::int2 LOCATION)
message(STATUS "Libint2 location: ${LIBINT2_LOC}")

