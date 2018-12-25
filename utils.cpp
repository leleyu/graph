//
// Created by leleyu on 2018-12-25.
//


#include <torch/torch.h>
#include "graph/utils.h"
#include <string.h>
#include <stdio.h>
#include <sstream>
#include <iostream>
#include <vector>


namespace graph {
  namespace utils {
    torch::data::Example<> parseLibSVM(std::string str) {

      std::vector<std::string> strings;
      std::istringstream f(str);
      string s, t;

      // label
      getline(f, s, ' ');
      auto label = at::tensor(std::stoi(s));
      std::vector<int> keys;
      std::vector<float> vals;

      while (getline(f, s, ' ')) {
        std::istringstream kv(s);
        getline(kv, t, ':');
        keys.push_back(std::stoi(t));
        getline(kv, t, ':');
        vals.push_back(std::stof(t));
      }

      int size = keys.size();
      auto indices = at::zeros({1, size}, at::TensorOptions().dtype(at::kLong));
      for (size_t i = 0; i < size; i++) indices[0][i] = keys[i];
      auto values  = at::zeros(size, at::TensorOptions().dtype(at::kFloat));
      for (size_t i = 0; i < size; i++) values[i] = vals[i];

      auto data = at::sparse_coo_tensor(indices, values);
      return {data, label};
    }
  };
};
