#include <iostream>
#include "torch/torch.h"


struct LogisticRegression: torch::nn::Module {
  
  LogisticRegression(int n_dim) {
    fc = register_module("fc1", torch::nn::Linear(n_dim, 1));
  }
  
  torch::Tensor forward(torch::Tensor x) {
    x = torch::sigmoid(fc->forward(x));
    return x;
  }

private:
  torch::nn::Linear fc{nullptr};
  
};


int main() {
  std::cout << "Hello, World!" << std::endl;
  return 0;
}