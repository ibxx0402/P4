cmake_minimum_required(VERSION 3.10)
project(img_server VERSION 1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Find dependencies
find_package(Threads REQUIRED)

find_package(OpenCV REQUIRED)

# Source files - explicitly set source files for now
set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    # Add more source files here as needed
)

# Include directories
include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})

# Define main executable with explicit sources
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

# Compiler flags
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${PROJECT_NAME} PRIVATE -g -O0 -Wall -Wextra)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -O2)
endif()

# Installation
install(TARGETS ${PROJECT_NAME} DESTINATION bin)

