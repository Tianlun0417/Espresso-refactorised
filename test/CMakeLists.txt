# test files
set(ESP_RE_LIBS espresso_refactorised m ${BLAS_LIBRARIES})

#add_executable(espresso_refactorised_main mnist_mlp.c)
#target_link_libraries(espresso_refactorised_main ${ESP_RE_LIBS} cblas)

#add_executable(espresso_refactorised_main read_mnist.c)
#target_link_libraries(espresso_refactorised_main ${ESP_RE_LIBS} cblas)

#add_executable(espresso_refactorised_main cifar10_cnn.c)
#target_link_libraries(espresso_refactorised_main ${ESP_RE_LIBS} cblas)

#add_executable(espresso_refactorised_main mnist_single_layer.c)
#target_link_libraries(espresso_refactorised_main ${ESP_RE_LIBS} cblas)

#add_executable(espresso_refactorised_main cblas_test.c)
#target_link_libraries(espresso_refactorised_main ${ESP_RE_LIBS} cblas)

#add_executable(espresso_refactorised_main int_tensor_test.c)
#target_link_libraries(espresso_refactorised_main ${ESP_RE_LIBS} cblas)

add_executable(espresso_refactorised_main AlexNet.c)
target_link_libraries(espresso_refactorised_main ${ESP_RE_LIBS} cblas)

#add_executable(espresso_refactorised_main ResNet_test.c)
#target_link_libraries(espresso_refactorised_main ${ESP_RE_LIBS} cblas)

#add_executable(espresso_refactorised_main VGG-11.c)
#target_link_libraries(espresso_refactorised_main ${ESP_RE_LIBS} cblas)