set(ESP_RE_LIBS espresso_refactorised m ${BLAS_LIBRARIES})

add_executable(espresso_refactorised_main AlexNet.c)
target_link_libraries(espresso_refactorised_main ${ESP_RE_LIBS} cblas)