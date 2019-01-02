//
// Created by leleyu on 2018-12-26.
//

#ifndef TEST_MODEL_H
#define TEST_MODEL_H

#include <torch/torch.h>

using namespace torch;

struct LogisticRegression: nn::Module {
public:

  explicit LogisticRegression(size_t n_dim) {
    fc1 = register_module("fc1", nn::Linear(n_dim, 10));
    fc2 = register_module("fc2", nn::Linear(10, 1));
//    weight = register_parameter("weight", torch::rand({1, (int) n_dim}));
//    bias = register_parameter("bias", torch::rand(1));
  }

  Tensor forward(Tensor x) {
//    x = torch::sigmoid(torch::_sparse_mm(x, weight.t()));
    x = relu(fc1->forward(x));
    x = sigmoid(fc2->forward(x));
    return x;
  }

private:
  torch::nn::Linear fc1{nullptr};
  torch::nn::Linear fc2{nullptr};
//  torch::Tensor weight;
//  torch::Tensor bias;
};


struct MeanAggregator: nn::Module {
public:
  explicit MeanAggregator(size_t input_dim, size_t output_dim) {
    auto options = nn::LinearOptions(input_dim, output_dim);
    options.with_bias_ = false;

    fc_nodes = register_module("fc_nodes", nn::Linear(options));
    fc_neibours = register_module("fc_neibours", nn::Linear(options));
  }

  /**
   * Aggregates features from neibours
   * @param nodes: features for nodes in current batch
   * @param neibours: features for neibours in current batch
   * @return: the aggregated feature
   */
  Tensor forward(const Tensor& nodes, const Tensor& neibours) {
    auto aggr_neibours = neibours.mean(1);

  }

private:
  nn::Linear fc_nodes{nullptr}, fc_neibours{nullptr};
};

struct GraphSage : torch::nn::Module {
public:
  explicit GraphSage() {

  }

  torch::Tensor forward(torch::Tensor x) {
    return x;
  }
};


#endif //TEST_MODEL_H
