//
// Created by leleyu on 2019-05-05.
//

#ifndef PYTORCH_LR_H
#define PYTORCH_LR_H

#include <iostream>

#include <torch/torch.h>
#include <torch/script.h>

namespace angel {

class LogisticRegression {
 public:
  explicit LogisticRegression(const std::string &path);

  at::Tensor forward(std::vector<torch::jit::IValue> inputs);

  void backward(at::Tensor batch_size,
                at::Tensor index, at::Tensor feats,
                at::Tensor values, at::Tensor bias,
                at::Tensor weights, at::Tensor targets);

 private:
  std::shared_ptr<torch::jit::script::Module> module_;
};

} // namespace angel


#endif //PYTORCH_LR_H
