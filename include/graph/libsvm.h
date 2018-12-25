//
// Created by leleyu on 2018-12-25.
//

#ifndef TEST_LIBSVMDATASET_H
#define TEST_LIBSVMDATASET_H

#include <fstream>

#include <torch/torch.h>
#include "graph/utils.h"

struct LibsvmDataset : torch::data::datasets::Dataset<LibsvmDataset> {

  explicit LibsvmDataset(std::string path, size_t n_dim) {
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
      auto example = graph::utils::parseLibSVM(line, n_dim);
      examples.push_back(example);
    }

    std::cout << examples.size() << std::endl;
  }

  torch::data::Example<> get(size_t index) override {
    return examples[index];
  }

  torch::optional<size_t> size() const override {
    return examples.size();
  }

private:
  std::vector<torch::data::Example<>> examples;
};

#endif //TEST_LIBSVMDATASET_H
