cmake_minimum_required(VERSION 3.17)
project(final_proj_cuda CUDA)

set(CMAKE_CUDA_STANDARD 11)

add_executable(final_proj_cuda main.cu)
#set_property(TARGET final_proj_cuda PROPERTY CUDA_ARCHITECTURES 61) # for 1070
#set_property(TARGET final_proj_cuda PROPERTY CUDA_ARCHITECTURES 70) # for v100

set_target_properties(
        final_proj_cuda
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)