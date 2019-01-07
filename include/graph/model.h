//
// Created by leleyu on 2018-12-26.
//

#ifndef TEST_MODEL_H
#define TEST_MODEL_H

#include <torch/torch.h>
#include <graph/mean.h>
#include <graph/dataset.h>

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

};


struct MeanAggregator0: nn::Module {
public:
  MeanAggregator0() {}
  explicit MeanAggregator0(size_t input_dim, size_t output_dim) {
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
    // num_nodes = nodes.size(0), feature_dim = nodes.size(1)
    // num_nodes = neibours.size(0), number_neibours = neibours.size(1), feature_dim = neibours.size(2)

    // do mean aggregation
    auto aggr_neibours = neibours.mean(1);

    // combine
    auto list = torch::TensorList({fc_nodes->forward(nodes), fc_neibours->forward(neibours)});
    auto out = relu(torch::cat(list));
    return out;
  }

private:
  nn::Linear fc_nodes{nullptr}, fc_neibours{nullptr};
};









#endif //TEST_MODEL_H
