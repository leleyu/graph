//
// Created by leleyu on 2019-05-05.
//

#include <angel/pytorch/lr.h>

namespace angel {

LogisticRegression::LogisticRegression(const std::string &path) {
  module_ = torch::jit::load(path);
//  std::cout << module_->is_optimized() << std::endl;
}

at::Tensor LogisticRegression::forward(std::vector<torch::jit::IValue> inputs) {
  return module_->forward(std::move(inputs)).toTensor();
}

void LogisticRegression::backward(at::Tensor batch_size,
                                  at::Tensor index, at::Tensor feats,
                                  at::Tensor values, at::Tensor bias,
                                  at::Tensor weights, at::Tensor targets) {
  std::vector<torch::jit::IValue> inputs;
  inputs.resize(6);
  inputs[0] = std::move(batch_size);
  inputs[1] = std::move(index);
  inputs[2] = std::move(feats);
  inputs[3] = std::move(values);
  inputs[4] = bias;
  inputs[5] = weights;

  auto outputs = module_->forward(inputs).toTensor();
  std::vector<torch::jit::IValue> loss_inputs;
  loss_inputs.resize(2);
  loss_inputs[0] = outputs;
  loss_inputs[1] = std::move(targets);

  auto loss = module_->get_method("loss")(loss_inputs).toTensor();
  loss.backward();

  std::cout << "bias_grad" << bias.grad().item().toFloat() << std::endl;

  std::cout << "loss=" << loss.item().toFloat() << std::endl;

//  bias.set_requires_grad(false);
//  weights.set_requires_grad(false);
//  bias.set_(bias.grad());
//  std::cout << "bias " << bias.item().toFloat() << std::endl;
//  weights.set_(weights.grad());
}


}

