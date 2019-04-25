//
// Created by leleyu on 2019-04-08.
//

#ifndef GRAPH_INTERFACE_COMMONS_H
#define GRAPH_INTERFACE_COMMONS_H

#define DEFINE_PRIMITIVE_ARRAY(_jarray) \
  void* _jarray##_cptr = env->GetPrimitiveArrayCritical(_jarray, &is_copy);

#define RELEASE_PRIMITIVE_ARRAY(_jarray) \
  env->ReleasePrimitiveArrayCritical(_jarray, _jarray##_cptr, 0);

#define DEFINE_MODEL_PTR(MODEL_TYPE, jptr) \
  auto* ptr = reinterpret_cast<MODEL_TYPE*>(jptr);

#define DEFINE_TORCH_TENSOR(_jarray, type) \
  auto _jarray##_option = torch::TensorOptions().dtype(type).requires_grad(false); \
  auto _jarray##_tensor = torch::from_blob(_jarray##_cptr, {env->GetArrayLength(_jarray)}, _jarray##_option);

#define DEFINE_JFLOATARRAY(_carray_ptr, len) \
  jfloatArray _carray_ptr##_jarray = env->NewFloatArray(len); \
  env->SetFloatArrayRegion(_carray_ptr##_jarray, 0, len, reinterpret_cast<float*>(_carray_ptr));

#define DEFINE_PRIMITIVE_ARRAYS4(jarray1, jarray2, jarray3, jarray4) \
  DEFINE_PRIMITIVE_ARRAY(jarray1);\
  DEFINE_PRIMITIVE_ARRAY(jarray2);\
  DEFINE_PRIMITIVE_ARRAY(jarray3);\
  DEFINE_PRIMITIVE_ARRAY(jarray4);

#define DEFINE_PRIMITIVE_ARRAYS5(jarray1, jarray2, jarray3, jarray4, jarray5) \
  DEFINE_PRIMITIVE_ARRAYS4(jarray1, jarray2, jarray3, jarray4); \
  DEFINE_PRIMITIVE_ARRAY(jarray5);

#define DEFINE_PRIMITIVE_ARRAYS6(jarray1, jarray2, jarray3, jarray4, jarray5, jarray6) \
  DEFINE_PRIMITIVE_ARRAYS5(jarray1, jarray2, jarray3, jarray4, jarray5); \
  DEFINE_PRIMITIVE_ARRAY(jarray6);

#define RELEASE_PRIMITIVE_ARRAYS4(jarray1, jarray2, jarray3, jarray4) \
  RELEASE_PRIMITIVE_ARRAY(jarray1);\
  RELEASE_PRIMITIVE_ARRAY(jarray2);\
  RELEASE_PRIMITIVE_ARRAY(jarray3);\
  RELEASE_PRIMITIVE_ARRAY(jarray4);

#define RELEASE_PRIMITIVE_ARRAYS5(jarray1, jarray2, jarray3, jarray4, jarray5) \
  RELEASE_PRIMITIVE_ARRAYS4(jarray1, jarray2, jarray3, jarray4) \
  RELEASE_PRIMITIVE_ARRAY(jarray5);

#define RELEASE_PRIMITIVE_ARRAYS6(jarray1, jarray2, jarray3, jarray4, jarray5, jarray6) \
  RELEASE_PRIMITIVE_ARRAYS5(jarray1, jarray2, jarray3, jarray4, jarray5);\
  RELEASE_PRIMITIVE_ARRAY(jarray6);

#define DEFINE_EMBEDDINGS(jinput_embeddings, embedding_dim) \
  auto length = env->GetArrayLength(jinput_embeddings); \
  assert(length % embedding_dim == 0); \
  auto size = length / embedding_dim; \
  auto input_embeddings = torch::from_blob(jinput_embeddings##_cptr, {size, embedding_dim});

#define DEFINE_GRAPH_STRUCTURE(jnodes, jneighbors, jmax_neighbor) \
  auto* cinodes = static_cast<int32_t*>(jnodes_##cptr); \
  auto* cineighbors = static_cast<int32_t*>(jneighbors_##cptr); \
  std::vector<int32_t> nodes(cinodes, cinodes + env->GetArrayLength(jnodes)); \
  std::vector<int32_t> neighbors(cineighbors, cineighbors + env->GetArrayLength(jneighbors)); \
  angel::graph::SubGraph sub_graph(nodes, neighbors, jmax_neighbor);

// define torch tensors

#define TORCH_OPTION_INT64 \
  (torch::TensorOptions().dtype(torch::kInt64).requires_grad(false))

#define TORCH_OPTION_INT32 \
  (torch::TensorOptions().dtype(torch::kInt32).requires_grad(false))

#define TORCH_OPTION_FLOAT \
  (torch::TensorOptions().requires_grad(false))

#define DEFINE_ZEROS_DIM2_INT64(_name, dim1, dim2) \
  auto _name = torch::zeros({dim1, dim2}, TORCH_OPTION_INT64);

#define DEFINE_ZEROS_DIM2_FLOAT(_name, dim1, dim2) \
  auto _name = torch::zeros({dim1, dim2}, TORCH_OPTION_FLOAT);

#define DEFINE_ZEROS_DIM1_INT64(_name, dim1) \
  auto _name = torch::zeros({dim1}, TORCH_OPTION_INT64);

#define DEFINE_ZEROS_DIM1_INT32(_name, dim1) \
  auto _name = torch::zeros({dim1}, TORCH_OPTION_INT32);

#define DEFINE_ACCESSOR_DIM1_INT64(_tensor) \
  auto _tensor##_acr = _tensor.accessor<int64_t, 1>();

#define DEFINE_ACCESSOR_DIM1_INT32(_tensor) \
  auto _tensor##_acr = _tensor.accessor<int32_t, 1>();

#define DEFINE_ACCESSOR_DIM2_INT64(_tensor) \
  auto _tensor##_acr = _tensor.accessor<int64_t, 2>();

#define DEFINE_ACCESSOR_DIM2_FLOAT(_tensor) \
  auto _tensor##_acr = _tensor.accessor<float, 2>();

#endif //GRAPH_INTERFACE_COMMONS_H
