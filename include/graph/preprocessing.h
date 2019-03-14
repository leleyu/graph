//
// Created by leleyu on 19-2-15.
//

#ifndef GRAPH_PREPROCESSING_H
#define GRAPH_PREPROCESSING_H

#include <algorithm>
#include <torch/torch.h>
#include <graph/graph.h>

namespace graph {
namespace preprocessing {

class LabelIndex {
 public:
  virtual torch::Tensor Transform(const NodeArray &nodes) const = 0;
  virtual torch::Tensor Loss(torch::Tensor y_pred, torch::Tensor y_true) const = 0;
  virtual void Insert(NodeId node, std::vector<Label> label) = 0;
  virtual void Load(const std::string& path) = 0;
  virtual double PrecisionScore(torch::Tensor y_pred, torch::Tensor y_true) const = 0;
  virtual double F1Score(torch::Tensor y_pred, torch::Tensor y_true, const std::string& option) const = 0;
};

class MultiLabelBinarizer: public LabelIndex {
 public:
  MultiLabelBinarizer() {
    num_class_ = -1;
  }

  virtual torch::Tensor Transform(const NodeArray &nodes) const override {
    int64_t num_node = nodes.size();
    auto binary = torch::zeros({num_node, num_class_});
    binary.set_requires_grad(false);
    auto f = binary.accessor<float, 2>();
    for (int64_t i = 0; i < num_node; i++) {
      NodeId node = nodes[i];
      auto &label = labels_[index_.find(node)->second];
      for (auto l : label)
        f[i][l] = 1.0f;
    }
    return binary;
  }

  virtual void Insert(NodeId node, std::vector<Label> label) override {
    index_[node] = static_cast<int32_t >(labels_.size());
    labels_.push_back(label);
    for (size_t i = 0; i < label.size(); i++)
      num_class_ = std::max(num_class_, label[i] + 1);
  }

  virtual void Load(const std::string& path) override {
    std::ifstream in(path);
    std::string line, c;

    while (getline(in, line)) {
      std::istringstream is(line);
      // NodeId
      getline(is, c, ' ');
      NodeId node_id = std::stoi(c);

      std::vector<Label> labels;
      while (getline(is, c, ' ')) {
        Label label = std::stoi(c);
        labels.push_back(label);
      }

      Insert(node_id, labels);
    }
  }
  virtual torch::Tensor Loss(torch::Tensor y_pred, torch::Tensor y_true) const override {
    auto loss =  -(y_true * torch::log_sigmoid(y_pred) + (1 - y_true) * torch::log_sigmoid(-y_pred));
    loss = loss.sum(1) / y_pred.size(1);
    return loss;
  }

  virtual double PrecisionScore(torch::Tensor y_pred, torch::Tensor y_true) const override {

  }

  virtual double F1Score(torch::Tensor y_pred, torch::Tensor y_true, const std::string& option) {

  }

 protected:
  std::vector<std::vector<Label>> labels_;
  IndexLookupTable index_;
  Label num_class_;
};

class MultiLabelIndex : public MultiLabelBinarizer {
 public:
  virtual torch::Tensor Transform(const NodeArray& nodes) const override {
    int64_t num_node = nodes.size();
    auto index = torch::zeros({num_node});
    auto f = index.accessor<float, 1>();
    for (size_t i = 0; i < num_node; i++) {
      NodeId node = nodes[i];
      const std::vector<Label>& labels = labels_[index_.find(node)->second];
      for (auto l : labels)
        if (l == current_class_)
          f[i] = 1.0f;
    }

    return index;
  }

  void SetCurrentClass(Label current_class) {
    current_class_ = current_class;
  }

  virtual torch::Tensor Loss(torch::Tensor y_pred, torch::Tensor y_true) const override {
    torch::Tensor undefined;
    y_pred = y_pred.squeeze();
//    std::cout << y_pred.sizes() << " " << y_pred.dtype() << std::endl;
//    std::cout << y_true.sizes() << " " << y_true.dtype() << std::endl;
    return torch::binary_cross_entropy_with_logits(y_pred, y_true, undefined, undefined, 1);
  }

  virtual double PrecisionScore(torch::Tensor y_pred, torch::Tensor y_true) const override {
    auto y = (y_pred.squeeze() >= 0.5).to(torch::kInt64);
//    std::cout << y.sizes() << std::endl;
//    std::cout << y_true.sizes() << std::endl;
    auto eq = y.eq(y_true.to(torch::kInt64));
//    std::cout << eq.sizes() << std::endl;
//    std::cin.get();
    auto n_right = eq.sum().item().toInt();
    auto n_total = y_pred.size(0);
//    std::cout << n_right << " " << n_total << std::endl;
    return n_right * 1.0 / n_total;
  }

  virtual double F1Score(torch::Tensor y_pred, torch::Tensor y_true, const std::string& option) const override {
    int n_class = y_pred.size(1);
    auto y = y_pred.argmax(1).to(torch::TensorOptions().dtype(torch::kInt64));
    int64_t true_negatives = 0;
    int64_t true_positives = 0;
    int64_t false_negatives = 0;
    int64_t false_positives = 0;
    double precision = 0;
    double recall = 0;
    double average_f1score = 0;
    for (int64_t i = 0; i < n_class; i++)
    {
      auto this_class_mask = torch::full(y.sizes(), i, torch::TensorOptions().dtype(torch::kInt64));
      auto positive_label_mask = y_true.eq(this_class_mask).to(torch::TensorOptions().dtype(torch::kInt64));
      auto mask = torch::ones(positive_label_mask.sizes(), torch::TensorOptions().dtype(torch::kInt64));
      auto negative_label_mask = torch::sub(mask, positive_label_mask);

      auto correct_null_predictions = torch::mul(torch::ne(y, this_class_mask).to(torch::TensorOptions().dtype(torch::kInt64)), negative_label_mask);
      double tmp_true_negatives =  torch::mul(correct_null_predictions, mask).sum().item().toInt();
      true_negatives += tmp_true_negatives;
      //std::cout << "true_neg=" << tmp_true_negatives << std::endl;

      auto correct_non_null_predictions = torch::mul(torch::eq(y, this_class_mask).to(torch::TensorOptions().dtype(torch::kInt64)), positive_label_mask);
      double tmp_true_positives =  torch::mul(correct_non_null_predictions, mask).sum().item().toInt();
      true_positives += tmp_true_positives;
      //std::cout << "true_pos=" << tmp_true_positives << std::endl;

      auto incorrect_null_predictions = torch::mul(torch::ne(y, this_class_mask).to(torch::TensorOptions().dtype(torch::kInt64)), positive_label_mask);
      double tmp_false_negatives =  torch::mul(incorrect_null_predictions, mask).sum().item().toInt();
      false_negatives += tmp_false_negatives;
      //std::cout << "false_neg=" << tmp_false_negatives << std::endl;

      auto incorrect_non_null_predictions = torch::mul(torch::eq(y, this_class_mask).to(torch::TensorOptions().dtype(torch::kInt64)), negative_label_mask);
      double tmp_false_positives =  torch::mul(incorrect_non_null_predictions, mask).sum().item().toInt();
      false_positives += tmp_false_positives;
      //std::cout << "false_pos=" << tmp_false_positives << std::endl;

      if (option == "macro") {
        precision = double(tmp_true_positives) / (tmp_true_positives + tmp_false_positives + 1e-13);
        recall = double(tmp_true_positives) / (tmp_true_positives + tmp_false_negatives + 1e-13);
        average_f1score +=  2 * recall * precision / (recall + precision + 1e-13);
      }
    }
    if (option == "macro") {
      return average_f1score;
    }
    else {
      precision = double(true_positives) / (true_positives + false_positives + 1e-13);
      recall = double(true_positives) / (true_positives + false_negatives + 1e-13);
      return  2 * recall * precision / (recall + precision + 1e-13);
    }
  }

 private:
  Label current_class_;
};

class SingleLabelIndex : public LabelIndex {
 public:
  virtual torch::Tensor Transform(const NodeArray& nodes) const override {
    int64_t num_node = nodes.size();
    auto index = torch::zeros({num_node}, torch::TensorOptions().dtype(torch::kInt64));
    auto f = index.accessor<int64_t, 1>();
    for (size_t i = 0; i < num_node; i++)
      f[i] = labels_.find(nodes[i])->second;
    return index;
  }

  virtual void Insert(NodeId node, std::vector<Label> label) override {
    assert(label.size() == 1);
    labels_[node] = label[0];
  }

  virtual void Load(const std::string& path) override {
    std::ifstream in(path);
    std::string line, c;

    while (getline(in, line)) {
      std::istringstream is(line);
      // NodeId
      getline(is, c, ' ');
      NodeId node_id = std::stoi(c);

      // label
      getline(is, c, ' ');
      Label label = std::stoi(c);

      labels_[node_id] = label;
    }
  }

  virtual torch::Tensor Loss(torch::Tensor y_pred, torch::Tensor y_true) const override {
    return torch::nll_loss(log_softmax(y_pred, 1), y_true);
  }

  virtual double PrecisionScore(torch::Tensor y_pred, torch::Tensor y_true) const override {
    auto y = y_pred.argmax(1).to(torch::TensorOptions().dtype(torch::kInt64));
    int n_right = torch::eq(y, y_true).sum().item().toInt();
    int n_total = y_pred.size(0);
    return n_right * 1.0 / n_total;
  }

  virtual double F1Score(torch::Tensor y_pred, torch::Tensor y_true, const std::string& option) const override {
    int n_class = y_pred.size(1);
    auto y = y_pred.argmax(1).to(torch::TensorOptions().dtype(torch::kInt64));
    int64_t true_negatives = 0;
    int64_t true_positives = 0;
    int64_t false_negatives = 0;
    int64_t false_positives = 0;
    double precision = 0;
    double recall = 0;
    double average_f1score = 0;
    for (int64_t i = 0; i < n_class; i++)
    {
      auto this_class_mask = torch::full(y.sizes(), i, torch::TensorOptions().dtype(torch::kInt64));
      auto positive_label_mask = y_true.eq(this_class_mask).to(torch::TensorOptions().dtype(torch::kInt64));
      auto mask = torch::ones(positive_label_mask.sizes(), torch::TensorOptions().dtype(torch::kInt64));
      auto negative_label_mask = torch::sub(mask, positive_label_mask);

      auto correct_null_predictions = torch::mul(torch::ne(y, this_class_mask).to(torch::TensorOptions().dtype(torch::kInt64)), negative_label_mask);
      double tmp_true_negatives =  torch::mul(correct_null_predictions, mask).sum().item().toInt();
      true_negatives += tmp_true_negatives;
      //std::cout << "true_neg=" << tmp_true_negatives << std::endl;

      auto correct_non_null_predictions = torch::mul(torch::eq(y, this_class_mask).to(torch::TensorOptions().dtype(torch::kInt64)), positive_label_mask);
      double tmp_true_positives =  torch::mul(correct_non_null_predictions, mask).sum().item().toInt();
      true_positives += tmp_true_positives;
      //std::cout << "true_pos=" << tmp_true_positives << std::endl;

      auto incorrect_null_predictions = torch::mul(torch::ne(y, this_class_mask).to(torch::TensorOptions().dtype(torch::kInt64)), positive_label_mask);
      double tmp_false_negatives =  torch::mul(incorrect_null_predictions, mask).sum().item().toInt();
      false_negatives += tmp_false_negatives;
      //std::cout << "false_neg=" << tmp_false_negatives << std::endl;

      auto incorrect_non_null_predictions = torch::mul(torch::eq(y, this_class_mask).to(torch::TensorOptions().dtype(torch::kInt64)), negative_label_mask);
      double tmp_false_positives =  torch::mul(incorrect_non_null_predictions, mask).sum().item().toInt();
      false_positives += tmp_false_positives;
      //std::cout << "false_pos=" << tmp_false_positives << std::endl;

      if (option == "macro") {
        precision = double(tmp_true_positives) / (tmp_true_positives + tmp_false_positives + 1e-13);
        recall = double(tmp_true_positives) / (tmp_true_positives + tmp_false_negatives + 1e-13);
        average_f1score +=  2 * recall * precision / (recall + precision + 1e-13);
      }
    }
    if (option == "macro") {
      return average_f1score;
    }
    else {
      precision = double(true_positives) / (true_positives + false_positives + 1e-13);
      recall = double(true_positives) / (true_positives + false_negatives + 1e-13);
      return  2 * recall * precision / (recall + precision + 1e-13);
    }
  }

 private:
  std::unordered_map<NodeId, Label> labels_;
};

} // namespace preprocessing
} // namespace graph

#endif //GRAPH_PREPROCESSING_H
