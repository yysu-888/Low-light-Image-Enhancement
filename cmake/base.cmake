Function (add_my_lib_include arg)
    find_package(OpenCV 4.0 REQUIRED)

    target_include_directories(${arg}
        PUBLIC 
            ${OpenCV_INCLUDE_DIRS}

            ${CMAKE_CURRENT_SOURCE_DIR}/include/
            ${CMAKE_CURRENT_SOURCE_DIR}/include/eigen-3.4.0)
    target_link_libraries(${arg}  
        PUBLIC 
        ${OpenCV_LIBS}
    )
    target_compile_definitions(${arg}
        PUBLIC
            HAVE_EIGEN
    )
endFunction()