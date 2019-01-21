//
// Created by leleyu on 19-1-14.
//

#ifndef GRAPH_METRIC_H
#define GRAPH_METRIC_H

#include <torch/torch.h>

namespace graph {
namespace metric {

namespace th = torch;

double precision_score(th::Tensor y_pred, th::Tensor y_true);

double roc_auc_score(th::Tensor y_pred, th::Tensor y_true);

double f1_score(th::Tensor y_pred, th::Tensor y_true, const std::string& type = "micro");
} // metric
} // graph

#endif //GRAPH_METRIC_H
