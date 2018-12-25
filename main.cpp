#include <iostream>
#include "torch/torch.h"
#include "graph/utils.h"
#include "graph/libsvm.h"


struct LogisticRegression: torch::nn::Module {
public:

  explicit LogisticRegression(size_t n_dim) {
    fc = register_module("fc1", torch::nn::Linear(n_dim, 1));
  }
  
  torch::Tensor forward(torch::Tensor x) {
    x = torch::sigmoid(fc->forward(x));
    return x;
  }

private:
  torch::nn::Linear fc{nullptr};
};



void train(const std::string input, size_t n_dim, size_t batch_size) {

  auto dataset = LibsvmDataset(input, n_dim);
  auto options = torch::data::DataLoaderOptions(batch_size);
  auto samplers = torch::data::samplers::SequentialSampler(dataset.size().value());
  auto data_loader = torch::data::make_data_loader(dataset, options, samplers);


  LogisticRegression lr(n_dim);

  // Instantiate an Adam optimizer algorithm to update the parameters
  torch::optim::SGD optimizer(lr.parameters(), 0.01);

  for (size_t epoch = 1; epoch <= 10; ++epoch) {
    float loss_sum = 0.0;
    int n_right = 0;
    int n_total = 0;
    for (auto batch : *data_loader) {
      optimizer.zero_grad();
      auto data = batch[0].data;
      auto label = batch[0].target;
      auto prediction = lr.forward(data);
      if (prediction.item().toFloat() >= 0.5 and label.item().toFloat() >= 0.5)
        n_right ++;
      if (prediction.item().toFloat() < 0.5 and label.item().toFloat() < 0.5)
        n_right ++;
      auto loss = torch::binary_cross_entropy(prediction, label);
      loss.backward();
      optimizer.step();
      loss_sum += loss.item().toFloat();
      n_total ++;
    }
    
    std::cout << "epoch=" << epoch << " loss=" << loss_sum / n_total << std::endl;
    std::cout << "precision=" << n_right * 1.0 / n_total << std::endl;
  }
}

int main() {
  std::string input = "../data/a9a_123d_train.libsvm";
  size_t n_dim = 124;
  train(input, n_dim, 1);
  return 0;
}
