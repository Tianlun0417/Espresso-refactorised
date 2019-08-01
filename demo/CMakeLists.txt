set(ESP_RE_LIBS espresso_refactorised m)

add_executable(espresso_refactorised_main BPAlexNet_demo.c)
target_link_libraries(espresso_refactorised_main ${ESP_RE_LIBS} openblas)

#add_executable(espresso_refactorised_main BPDenseNet_demo.c)
#target_link_libraries(espresso_refactorised_main ${ESP_RE_LIBS} openblas)

#add_executable(espresso_refactorised_main BPResNet_demo.c)
#target_link_libraries(espresso_refactorised_main ${ESP_RE_LIBS} openblas)

#add_executable(espresso_refactorised_main AlexNet_demo.c)
#target_link_libraries(espresso_refactorised_main ${ESP_RE_LIBS} openblas)

#add_executable(espresso_refactorised_main ResNet_demo.c)
#target_link_libraries(espresso_refactorised_main ${ESP_RE_LIBS} openblas)

#add_executable(espresso_refactorised_main VGG_demo.c)
#target_link_libraries(espresso_refactorised_main ${ESP_RE_LIBS} openblas)

#add_executable(espresso_refactorised_main SqueezeNet_demo.c)
#target_link_libraries(espresso_refactorised_main ${ESP_RE_LIBS} openblas)

#add_executable(espresso_refactorised_main DenseNet_demo.c)
#target_link_libraries(espresso_refactorised_main ${ESP_RE_LIBS} openblas)

#add_executable(espresso_refactorised_main AlexNet_demo.c, ResNet_demo.c, VGG_demo.c, SqueezeNet_demo.c)
#target_link_libraries(espresso_refactorised_main ${ESP_RE_LIBS} openblas)