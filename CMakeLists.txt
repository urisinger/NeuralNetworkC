
cmake_minimum_required (VERSION 3.8)

project ("NeuralNetwork")

# Add source to this project's executable.
include_directories(headers)
add_executable (NeuralNetwork "src/Main.c" "src/LinearAlgebra.c" "headers/LinearAlgebra.h" "src/Layer.c" "headers/Layer.h")
target_link_libraries(NeuralNetwork PRIVATE m)

# TODO: Add tests and install targets if needed.
