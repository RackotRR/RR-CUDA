cmake_minimum_required(VERSION 3.20)
project(RRCUDA)

include(CheckLanguage)
check_language(CUDA)

option(RR_CUDA_ARCH "CUSTOM CUDA ARCHITECTURE")

if (CMAKE_CUDA_COMPILER)
    message("CUDA compiler found. Enable RRCUDA module")
    enable_language(CUDA)
    add_library(RRCUDA INTERFACE)
    set_target_properties(RRCUDA
        PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    )
    target_include_directories(RRCUDA
        INTERFACE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    if (${RR_CUDA_ARCH} STREQUAL "OFF")
        message("RR_CUDA_ARCH not provided. Default to native")
        set_property(TARGET RRCUDA PROPERTY CUDA_ARCHITECTURES native)
    else()
        message("RR_CUDA_ARCH is set: ${RR_CUDA_ARCH}")
        set_property(TARGET RRCUDA PROPERTY CUDA_ARCHITECTURES ${RR_CUDA_ARCH})
    endif()

else()
    message("CUDA compiler not found. Disable RRCUDA module")
endif()