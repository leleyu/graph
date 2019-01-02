//
// Created by leleyu on 2018-12-25.
//

#include "torch/data.h"
#include "torch/torch.h"
#include "graph/model.h"

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

void test_mm() {
  auto x = torch::ones({1, 10});
//  auto y = torch::ones({3, 5});
//  auto z = torch::mm(x, y.t());

//  auto xs = x.to_sparse();

//  std::cout << xs << std::endl;

//  std::cout << xs.dim() << std::endl;
//  std::cout << xs.size(0) << std::endl;

//  std::cout << torch::_sparse_mm(xs, x.t()) << std::endl;

  auto indices = at::zeros({2, 5}, torch::TensorOptions().dtype(torch::kLong));
  for (int a = 0; a < 5; a ++) indices[0][a] = 0;
  for (int a = 0; a < 5; a ++) indices[1][a] = a;
//
  auto values  = at::ones(5);
  auto sparse = torch::sparse_coo_tensor(indices, values, {1, 10});

  std::cout << sparse << std::endl;
  std::cout << torch::_sparse_mm(sparse, x.t()) << std::endl;


}

void test_parameters() {
  LogisticRegression lr(10);
  auto parameters = lr.parameters();
  for (int i = 0; i < parameters.size(); i ++) {
    auto & v = parameters[i];
    std::cout << v << std::endl;
  }
}

void test_sparse_gradient() {
  auto options = torch::TensorOptions().requires_grad(true);
  auto a = torch::ones({2, 4}, options);
  auto b = a.to_sparse();

  auto c = torch::_sparse_mm(b, a.t());
  std::cout << c << std::endl;

  c.backward();

  std::cout << a.grad() << std::endl;

}

void test_random() {
  auto a = torch::rand({10, 2});
  std::cout << a << std::endl;
}

void test_view() {
  auto a = torch::zeros({1, 10});
  auto b = a.view({2, 3, -1});
  std::cout << b << std::endl;
}

void test_mean() {
  auto a = torch::ones({2, 10});
  auto b = a.mean({1});
  auto c = a.sum({1});
  std::cout << b << std::endl;
  std::cout << c << std::endl;

  auto d = torch::ones({3, 2, 10});
  auto e = d.mean(1);
  std::cout << d << std::endl;
  std::cout << e << std::endl;
}

void test_embedding() {
  auto e = torch::nn::Embedding(10, 20);
  auto weight = e.get()->weight;
  for (int i = 0; i < 10; i ++)
    for (int j = 0; j < 20; j ++)
      weight[i][j] = i;

  std::cout << e.get() << std::endl;
  auto indices = torch::zeros({2}, TensorOptions().dtype(torch::kInt64));
  indices[0] = 0; indices[1] = 5;
  auto r = e.get()->forward(indices);
  std::cout << r << std::endl;
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
//  test_loader();
//  test_mm();
//  test_parameters();
//  test_sparse_gradient();
//  test_random();
//  test_view();
  test_mean();
//  test_embedding();
  return 0;
}

