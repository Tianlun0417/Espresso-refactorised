# test files
set(ESP_RE_LIBS espresso_refactorised m ${BLAS_LIBRARIES})

#add_executable(espresso_refactorised_main mnist_mlp.c)
#target_link_libraries(espresso_refactorised_main ${ESP_RE_LIBS} cblas)

#add_executable(espresso_refactorised_main read_mnist.c)
#target_link_libraries(espresso_refactorised_main ${ESP_RE_LIBS} cblas)

add_executable(espresso_refactorised_main cifar10_cnn.c)
target_link_libraries(espresso_refactorised_main ${ESP_RE_LIBS} cblas)
#
#add_executable(espresso_refactorised_main mnist_single_layer.c)
#target_link_libraries(espresso_refactorised_main ${ESP_RE_LIBS} cblas)