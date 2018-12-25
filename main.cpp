#include <iostream>
#include "torch/torch.h"
#include "graph/utils.h"
#include "graph/libsvm.h"


struct LogisticRegression: torch::nn::Module {
public:

  explicit LogisticRegression(int n_dim) {
    fc = register_module("fc1", torch::nn::Linear(n_dim, 1));
  }
  
  torch::Tensor forward(torch::Tensor x) {
    x = torch::sigmoid(fc->forward(x));
    return x;
  }

private:
  torch::nn::Linear fc{nullptr};
};



void train(std::string input, int n_dim, int batch_size) {

  auto dataset = LibsvmDataset(input);
  auto options = torch::data::DataLoaderOptions(batch_size);
  auto samplers = torch::data::samplers::SequentialSampler(dataset.size().value());
  auto data_loader = torch::data::make_data_loader(dataset, options, samplers);


  LogisticRegression lr(n_dim);

  // Instantiate an Adam optimizer algorithm to update the parameters
  torch::optim::Adam optimizer(lr.parameters(), 0.1);

  for (size_t epoch = 1; epoch <= 10; ++epoch) {
    size_t batch_index = 0;

    for (auto batch : *data_loader) {
      
    }

  }
}

int main() {
  std::string input = "data/a9a_123d_train.libsvm";
  int n_dim = 123;
  train(input, n_dim, 10);
  return 0;
}
