file(GLOB_RECURSE __PPLNN_MODEL_ONNX_SRC__ src/ppl/nn/models/onnx/*.cc)
add_library(pplnn_onnx_static STATIC ${PPLNN_SOURCE_EXTERNAL_ONNX_MODEL_SOURCES} ${__PPLNN_MODEL_ONNX_SRC__})
unset(__PPLNN_MODEL_ONNX_SRC__)

target_compile_definitions(pplnn_onnx_static PUBLIC PPLNN_ENABLE_ONNX_MODEL)
target_link_libraries(pplnn_onnx_static PUBLIC pplnn_basic_static)

include(cmake/protobuf.cmake)
target_link_libraries(pplnn_onnx_static PUBLIC libprotobuf)
target_include_directories(pplnn_onnx_static PRIVATE ${protobuf_SOURCE_DIR}/src)

if(PPLNN_INSTALL)
    install(DIRECTORY include/ppl/nn/models/onnx DESTINATION include/ppl/nn/models)
    install(TARGETS pplnn_onnx_static DESTINATION lib)
endif()
