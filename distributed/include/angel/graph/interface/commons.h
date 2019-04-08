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

#define DEFINE_PRIMITIVE_ARRAYS5(jarray1, jarray2, jarray3, jarray4, jarray5) \
  DEFINE_PRIMITIVE_ARRAY(jarray1);\
  DEFINE_PRIMITIVE_ARRAY(jarray2);\
  DEFINE_PRIMITIVE_ARRAY(jarray3);\
  DEFINE_PRIMITIVE_ARRAY(jarray4);\
  DEFINE_PRIMITIVE_ARRAY(jarray5);

#define DEFINE_PRIMITIVE_ARRAYS6(jarray1, jarray2, jarray3, jarray4, jarray5, jarray6) \
  DEFINE_PRIMITIVE_ARRAYS5(jarray1, jarray2, jarray3, jarray4, jarray5); \
  DEFINE_PRIMITIVE_ARRAY(jarray6);

#define RELEASE_PRIMITIVE_ARRAYS5(jarray1, jarray2, jarray3, jarray4, jarray5) \
  RELEASE_PRIMITIVE_ARRAY(jarray1);\
  RELEASE_PRIMITIVE_ARRAY(jarray2);\
  RELEASE_PRIMITIVE_ARRAY(jarray3);\
  RELEASE_PRIMITIVE_ARRAY(jarray4);\
  RELEASE_PRIMITIVE_ARRAY(jarray5);

#define RELEASE_PRIMITIVE_ARRAYS6(jarray1, jarray2, jarray3, jarray4, jarray5, jarray6) \
  RELEASE_PRIMITIVE_ARRAYS5(jarray1, jarray2, jarray3, jarray4, jarray5);\
  RELEASE_PRIMITIVE_ARRAY(jarray6);

#define DEFINE_EMBEDDINGS(jself_embeddings, jneibor_embeddings, embedding_dim) \
  auto length = env->GetArrayLength(jself_embeddings); \
  assert(length % embedding_dim == 0); \
  assert(length == env->GetArrayLength(jneibor_embeddings)); \
  auto size = length / embedding_dim; \
  auto self_embeddings = torch::from_blob(jself_embeddings##_cptr, {size, embedding_dim}); \
  auto neibor_embeddings = torch::from_blob(jneibor_embeddings##_cptr, {size, embedding_dim});

#define DEFINE_GRAPH_STRUCTURE(jnodes, jneibors, jmax_neibor) \
  auto* cinodes = static_cast<int32_t*>(jnodes_##cptr); \
  auto* cineibors = static_cast<int32_t*>(jneibors_##cptr); \
  std::vector<int32_t> nodes(cinodes, cinodes + env->GetArrayLength(jnodes)); \
  std::vector<int32_t> neibors(cineibors, cineibors + env->GetArrayLength(jneibors)); \
  SubGraph sub_graph(nodes, neibors, jmax_neibor);

#endif //GRAPH_INTERFACE_COMMONS_H
