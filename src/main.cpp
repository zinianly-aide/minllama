#include "minllama.h"

#include <iostream>

int main(int argc, char **argv) {
    std::cout << "minllama: skeleton build, model inference not implemented yet\n";
    if (argc > 1) {
        std::cout << "requested model path: " << argv[1] << "\n";
    }
    return 0;
}
