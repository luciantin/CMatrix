cmake_minimum_required(VERSION 3.25)
project(CMatrix C)

set(CMAKE_C_STANDARD 11)
set(DEBUG false CACHE BOOL "Build in debug mode")
set(USES_EMCMAKE false CACHE BOOL "Use emcc link flags")

message("USES_EMCMAKE=${USES_EMCMAKE}")
message("DEBUG=${DEBUG}")

add_executable(CMatrix
        main.c
        helpers/nnc_matrix.h
        helpers/nnc_config.h
        helpers/nnc_matrix.c
        helpers/nnc_layer.c
        helpers/nnc_layer.h
        helpers/nnc_config.c
        helpers/nnc_datasets.h
        helpers/nnc_activation_layer.c
        helpers/nnc_activation_layer.h
        helpers/nnc_loss_function.c
        helpers/nnc_loss_function.h
        helpers/nnc_vector.c
        helpers/nnc_vector.h
        helpers/nnc_optimizer.c
        helpers/nnc_optimizer.h
        training/AutoGenTest.h
        training/AutoGenTest.c
        development.h
        helpers/nnc_model.c
        helpers/nnc_model.h
        helpers/nnc_serializer.h
        helpers/nnc_serializer.c
        helpers/nnc_trainer.c
        helpers/nnc_trainer.h
        helpers/nnc_statistics.c
        helpers/nnc_statistics.h
        helpers/nnc_list.c
        helpers/nnc_list.h
        helpers/nnc_string.c
        helpers/nnc_string.h)
#target_link_libraries(CMatrix PRIVATE m)

if( USES_EMCMAKE )
    set_target_properties(CMatrix PROPERTIES LINK_FLAGS "--minify 0 -Oz -g0 -flto --emrun")
    set(CMAKE_EXECUTABLE_SUFFIX ".html")
endif()

if( DEBUG )
    target_compile_definitions(CMatrix PRIVATE DEBUG=1)
else()
    target_compile_definitions(CMatrix PRIVATE DEBUG=0)
endif()