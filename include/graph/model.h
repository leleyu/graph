//
// Created by leleyu on 2018-12-26.
//

#ifndef GRAPH_MODEL_H
#define GRAPH_MODEL_H

#include <torch/torch.h>

#include <graph/layer.h>
#include <graph/dataset.h>

namespace graph {
namespace model {

namespace th = torch;

struct TwoLayerMLP : th::nn::Module {
public:
  explicit TwoLayerMLP(size_t n_input, size_t n_hidden, size_t n_output) {
    fc1 = register_module("fc1", th::nn::Linear(n_input, n_hidden));
    fc2 = register_module("fc2", th::nn::Linear(n_hidden, n_output));
  }

  th::Tensor forward(th::Tensor x) {
    x = relu(fc1->forward(x));
    x = fc2->forward(x);
    return x;
  }

private:
  torch::nn::Linear fc1{nullptr};
  torch::nn::Linear fc2{nullptr};

};

struct LogisticRegression : th::nn::Module {
public:
  explicit LogisticRegression(size_t n_input, size_t n_output) {
    fc = register_module("fc1", th::nn::Linear(n_input, n_output));
  }

  th::Tensor forward(th::Tensor x) {
    x = fc->forward(x);
    return x;
  }

private:
  torch::nn::Linear fc{nullptr};

};
} // namespace model
} // namespace graph


#endif //GRAPH_MODEL_H
