cmake_minimum_required(VERSION 3.25)
project(CMatrix C)

set(CMAKE_C_STANDARD 11)


add_executable(CMatrix main.c helpers/nnc_matrix.h helpers/nnc_config.h helpers/nnc_matrix.c helpers/nnc_layer.c helpers/nnc_layer.h helpers/nnc_config.c helpers/nnc_datasets.h helpers/nnc_activation_layer.c helpers/nnc_activation_layer.h helpers/nnc_loss_function.c helpers/nnc_loss_function.h helpers/nnc_vector.c helpers/nnc_vector.h helpers/nnc_optimizer.c helpers/nnc_optimizer.h AutoGenTest.h AutoGenTest.c development.h helpers/nnc_model.c helpers/nnc_model.h)
target_link_libraries(CMatrix PRIVATE m)

