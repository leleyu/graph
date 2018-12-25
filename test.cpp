//
// Created by leleyu on 2018-12-25.
//

#include "torch/data.h"
#include "torch/torch.h"

#include <vector>
#include <iostream>

using namespace torch::data;

struct DummyDataset : datasets::Dataset<DummyDataset, int> {
  explicit DummyDataset(size_t size = 100) : size_(size) {}

  int get(size_t index) override {
    return 1 + index;
  }

  torch::optional<size_t> size() const override {
    return size_;
  }

  size_t size_;
};

struct ExampleDataset : datasets::Dataset<ExampleDataset> {
  explicit ExampleDataset(size_t size = 100) : size_(size) {}

  Example<> get(size_t index) override {
    return {torch::ones(4), torch::ones(1)};
  }

  torch::optional<size_t> size() const override {
    return size_;
  }

  size_t size_;
};


void tensor_sparse_test() {
  std::vector<int> keys = {1, 2, 3};
  std::vector<float> vals = {1.0, 2.0, 3.0};
  torch::Tensor k = torch::tensor(keys);
  torch::Tensor v = torch::tensor(vals);
  auto options = torch::TensorOptions()
    .dtype(torch::kFloat32)
    .layout(torch::kSparse)
    .device(torch::kCPU)
    .requires_grad(false);

  torch::Tensor x = torch::sparse_coo_tensor(k, v, options);
  std::cout << x << std::endl;
}

void tensor_dense_test() {
  torch::Tensor d = torch::rand({2, 3});
  std::cout << d.dim() << std::endl;

  std::cout << d.size(0) << std::endl;
  float* data = reinterpret_cast<float*>(d.data_ptr());
  for (size_t i = 0; i < 6; i++)
    std::cout << data[i] << std::endl;

  std::cout << d << std::endl;
}

void test_sampler() {
  auto sampler = samplers::SequentialSampler(100);
  size_t batch_size = 15;

  auto dataset = DummyDataset(100);
  while (true) {
    auto indices = sampler.next(batch_size);
    if (indices.has_value()) {
      auto batch = dataset.get_batch(indices.value());
      std::cout << batch << std::endl;
    } else {
      break;
    }
  }
}

void test_cat_tensors() {
  std::vector<torch::Tensor> tensors;
  tensors.push_back(torch::zeros({1, 10}));
  tensors.push_back(torch::ones({1, 10}));
  torch::TensorList list(tensors.data(), tensors.size());
  for (auto tensor : list) {
    std::cout << tensor << std::endl;
  }
  std::cout << torch::cat(list, 0) << std::endl;
  std::cout << torch::stack(list, 0) << std::endl;
}

void test_sparse_tensor() {
  auto indices = at::zeros({1, 5}, torch::TensorOptions().dtype(torch::kLong));
  indices[0][0] = 0;
  indices[0][1] = 1;
  indices[0][2] = 2;
  indices[0][3] = 3;
  indices[0][4] = 4;
  std::cout << indices.dim() << std::endl;
  std::cout << indices << std::endl;

  auto values  = at::ones(5);
  std::cout << values << std::endl;

  auto example = at::sparse_coo_tensor(indices, values);
  std::cout << example << std::endl;
}


void test_loader() {
  auto options = DataLoaderOptions(15);
  auto samplers = torch::data::samplers::SequentialSampler(100);
  auto data_loader = torch::data::make_data_loader(ExampleDataset(), options, samplers);
  std::cout << data_loader->options().batch_size << std::endl;

  for (auto batch : *data_loader) {
//    std::cout << batch << std::endl;

  }

}


int main() {
//  DummyDataset d;
//  std::vector<int> batch = d.get_batch({0, 1, 2, 3, 4});
//  for (auto i : batch)
//    std::cout << i << std::endl;

//  tensor_dense_test();
//  test_sampler();
//  test_cat_tensors();
//  test_sparse_tensor();
  test_loader();
  return 0;
}

