cmake_minimum_required(VERSION 3.9)

project(extract_frames)

list(APPEND CUDA_NVCC_FLAGS "-std=c++11")

if(NOT CMAKE_CUDA_FLAGS)
    set(CMAKE_CUDA_FLAGS "-cudart shared")
endif()

find_package(CUDA REQUIRED)

set(CUDA_LIBRARIES PUBLIC ${CUDA_LIBRARIES})

include_directories(../../include)

cuda_add_executable(${PROJECT_NAME}
    extract_frames.cpp
    cuda/utils.cu
    )

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)

#set_target_properties(${PROJECT_NAME} PROPERTIES
#    POSITION_INDEPENDENT_CODE ON
#    CUDA_SEPARABLE_COMPILATION ON
#    )

target_link_libraries(${PROJECT_NAME} PUBLIC nvvl)

set(WITH_OPENCV 1)

if (WITH_OPENCV)
    # If we are in here, they explicitly asked for it, so make it required
    find_package(OpenCV CONFIG REQUIRED COMPONENTS core cudaarithm cudaimgproc imgcodecs)
    target_compile_definitions(${PROJECT_NAME} PRIVATE HAVE_OPENCV=1)
    target_link_libraries(${PROJECT_NAME} PUBLIC opencv_core PRIVATE ${OpenCV_LIBS})
endif()

#add_custom_target(examples DEPENDS extract_frames)
