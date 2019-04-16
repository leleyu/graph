//
// Created by leleyu on 19-1-14.
//

#ifndef GRAPH_METRIC_H
#define GRAPH_METRIC_H

#include <torch/torch.h>

namespace graph {
namespace metric {

double PrecisionScore(torch::Tensor y_pred, torch::Tensor y_true);

double AucScore(torch::Tensor y_pred, torch::Tensor y_true);

double F1Score(torch::Tensor y_pred, torch::Tensor y_true, const std::string& type = "micro");
} // metric
} // graph

#endif //GRAPH_METRIC_H
