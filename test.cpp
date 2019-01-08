//
// Created by leleyu on 2018-12-25.
//

#include "torch/data.h"
#include "torch/torch.h"
#include "graph/model.h"
#include "graph/dataset.h"

#include <vector>
#include <iostream>
#include <unordered_map>

using namespace torch::data;
using namespace torch;

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
  std::cout << torch::cat(list, 1) << std::endl;

  torch::TensorList list2({torch::zeros({1, 10}), torch::ones({1, 10})});
  std::cout << torch::cat(list2, 1) << std::endl;
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

void test_embedding_indices() {
  auto features = torch::nn::Embedding(10, 5);
  auto weight = features.get()->weight;
  for (int i = 0; i < 10; i ++)
    for (int j = 0; j < 5; j ++)
      weight[i][j] = i;


  auto indices = torch::zeros({2, 4}, TensorOptions().dtype(torch::kInt64));
  for (int i = 0; i < 2; i ++)
    for (int j = 0; j < 4; j ++)
      indices[i][j] = j;

  auto r = features.get()->forward(indices);
  std::cout << r << std::endl;
}

void test_tensor_ref() {
  torch::Tensor t;
  std::cout << t.size(0) << std::endl;
}

void test_adj_dataset() {
  std::string path = "../data/cora/cora.adjs";
  graph::dataset::AdjList adj;
  graph::dataset::load_edges(path, &adj);

  std::cout << adj.starts.size() << std::endl;
  std::cout << adj.dsts.size() << std::endl;

  for (auto item: adj.src_to_index) {
    std::cout << "src=" << item.first << std::endl;
    int index = item.second;

    for (int i = adj.starts[index]; i < adj.starts[index + 1]; i ++) {
      std::cout << adj.dsts[i] << " ";
    }
    std::cout << std::endl;
    std::cin.get();
  }
}

void test_node_features() {
  std::string path = "../data/cora/cora.content.id";
  using namespace graph::dataset;
  Nodes nodes;
  load_features(path, &nodes, 1433, 2708);

  auto& features = nodes.features;
  auto& labels = nodes.labels;
  for (int i = 0; i < 2708; i ++) {
    std::cout << labels[i].item().toFloat() << std::endl;
    auto f = features[i];
    auto size = f.size(0);
    for (int j = 0; j < size; j ++)
      if (f[j].item().toFloat() > 0) std::cout << j << " ";
    std::cout << std::endl;
    std::cin.get();
  }
}

void test_tensor_assign() {
  auto a = torch::zeros({3, 5});
  a[0][0] = 1;

  std::vector<float> data = {0, 1, 2, 3, 4};
  data.resize(5);
  std::cout << data << std::endl;
//  memcpy(a.data_ptr(), static_cast<void*>(data.data()), 5);

  float* ptr = static_cast<float*>(a.data_ptr());

  memcpy(static_cast<void*>(ptr + 5), static_cast<void*>(data.data()), 5*sizeof(float));
//  ptr[0] = 0.1;
  std::cout << a << std::endl;

  a[2] = torch::ones({5});
  std::cout << a << std::endl;
}

void test_tensor_get() {
  auto x = torch::zeros({2}, torch::TensorOptions().dtype(torch::kInt32));
  float* ptr = static_cast<float*>(x.data_ptr());
  std::cout << ptr[0] << std::endl;
}

void test_unorderred_map() {
  std::unordered_map<int, int> map;
  map[1] = 0;
  std::cout << map[1] << std::endl;
  auto v = map.find(1);
  std::cout << v->second << std::endl;

  v = map.find(2);
  std::cout << (v == map.end()) << std::endl;
//  std::cout << v->second << std::endl;
}

void test_tensor_indices() {
  auto t = torch::ones({4, 10});
  t[0] = 2;
  std::cout << t << std::endl;
  auto indice = torch::ones({2}, TensorOptions().dtype(kInt32));
  std::cout << indice << std::endl;
  std::cout << indice.dim() << std::endl;
  int j = 0;

  std::cout << t[j] << std::endl;
}

void test_tensor_copy(Tensor x) {
  std::cout << x.use_count() << std::endl;
  auto y = x[0];
  std::cout << x.use_count() << std::endl;

}


void test_tensor_copy() {
  auto x = torch::randn({3, 5});
//  std::cout << x.use_count() << std::endl;
//  test_tensor_copy(x);
//  std::cout << x.use_count() << std::endl;

  auto y = x.relu();

  std::cout << x << std::endl;
  std::cout << y << std::endl;
}

torch::Tensor allocate() {
  return torch::ones({5});
}

void test_tensor_allocate() {
  auto a = allocate();
  std::cout << a << std::endl;
  std::cout << a.use_count() << std::endl;
}

void test_cross_entropy_loss() {
//  auto input = torch::randn({3, 5});
//  input.set_requires_grad(false);
//
//  auto target = torch::empty({3}, TensorOptions().dtype(kInt64));
//  target.random_(5);
//
//  std::cout << torch::softmax(input, 0) << std::endl;
//
//  auto output = torch::nll_loss(torch::log_softmax(input, 1), target);
//  std::cout << output << std::endl;

  auto input = torch::zeros({3, 5});
  std::vector<float> v = {-0.1414,  0.4857, -1.7201,  0.6922, -1.3067,
            0.4282, -0.7197,  0.1772, -1.0167,  0.7329,
            1.0204, -0.2235,  0.2687, -1.3497, -1.4210};
  memcpy(input.data_ptr(), v.data(), 15*sizeof(float));

  std::cout << input << std::endl;

  auto target = torch::empty({3}, TensorOptions().dtype(kInt64));
  target[0] = 3;
  target[1] = 4;
  target[2] = 0;

  auto output = torch::nll_loss(torch::log_softmax(input, 1), target);

  std::cout << output << std::endl;
}

void test_tensor_reshape() {
  auto t = torch::ones({10});
  std::cout << t << std::endl;
  std::cout << t.resize_({5}) << std::endl;
  t.reset();
  std::cout << t << std::endl;
}


int main() {
//  DummyDataset d;
//  std::vector<int> batch = d.get_batch({0, 1, 2, 3, 4});
//  for (auto i : batch)
//    std::cout << i << std::endl;

//  tensor_dense_test();
//  test_sampler();
// test_cat_tensors();
//  test_sparse_tensor();
//  test_loader();
//  test_mm();
//  test_parameters();
//  test_sparse_gradient();
//  test_random();
//  test_view();
//  test_mean();
//  test_embedding();
//  test_embedding_indices();
//  test_tensor_ref();
//  test_adj_dataset();
  // test_node_features();
//  test_tensor_assign();
//  test_tensor_get();
//  test_unorderred_map();
//  test_tensor_indices();
//  test_tensor_copy();
//  test_tensor_allocate();
//  test_cross_entropy_loss();
  test_tensor_reshape();
  return 0;
}

