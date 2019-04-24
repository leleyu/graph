//
// Created by leleyu on 2019-04-08.
//

#include <angel/graph/interface/supervised_graphsage_jni.h>
#include <angel/graph/interface/commons.h>
#include <angel/graph/model/graphsage.h>


/*
 * Class:     com_tencent_angel_graph_model_SupervisedGraphSage
 * Method:    backward
 * Signature: (J[F[F[II[I[I[I[Ljava/lang/String;)[[F
 */
JNIEXPORT jobjectArray JNICALL Java_com_tencent_angel_graph_model_SupervisedGraphSage_backward
  (JNIEnv *env, jobject jobj, jlong jptr,
    jfloatArray jinput_embeddings,
    jintArray jbatch,
    jint jmax_neibor,
    jintArray jnodes,
    jintArray jneibors,
    jintArray jtargets,
    jobjectArray jkeys) {

  jboolean is_copy;

  // Get primitive arrays.
  DEFINE_PRIMITIVE_ARRAYS5(jinput_embeddings, jbatch, jnodes, jneibors, jtargets);

  // model ptr
  DEFINE_MODEL_PTR(angel::graph::SupervisedGraphSage, jptr);
  int embedding_dim = ptr->GetDim();

  // embeddings
  DEFINE_EMBEDDINGS(jinput_embeddings, embedding_dim);

  // graph structures
  DEFINE_GRAPH_STRUCTURE(jnodes, jneibors, jmax_neibor);

  // input nodes
  DEFINE_TORCH_TENSOR(jbatch, torch::kInt64);
  DEFINE_TORCH_TENSOR(jtargets, torch::kF32);

  // Forward
  auto grads = ptr->Backward(jbatch_tensor, sub_graph, input_embeddings, jtargets_tensor);

  // Release them
  RELEASE_PRIMITIVE_ARRAYS5(jinput_embeddings, jbatch, jnodes, jneibors, jtargets);

  // create a two-dimensional array
  int n_keys = env->GetArrayLength(jkeys);

  jclass floatArrayClass = env->FindClass("[F");
  jobjectArray objects = env->NewObjectArray(n_keys, floatArrayClass, nullptr);

  for (int i = 0; i < n_keys; i++) {
    auto jstr = reinterpret_cast<jstring>(env->GetObjectArrayElement(jkeys, i));
    const char* jstr_ptr = env->GetStringUTFChars(jstr, &is_copy);
    auto grad = grads.find(std::string(jstr_ptr))->second;
    grad = grad.view({-1});
    auto grad_ptr = grad.data_ptr();
    DEFINE_JFLOATARRAY(grad_ptr, grad.size(0));
    env->SetObjectArrayElement(objects, i, grad_ptr_jarray);
    env->ReleaseStringUTFChars(jstr, jstr_ptr);
  }

  return objects;
}

/*
 * Class:     com_tencent_angel_graph_model_SupervisedGraphSage
 * Method:    initNetwork
 * Signature: (II[I)J
 */
JNIEXPORT jlong JNICALL Java_com_tencent_angel_graph_model_SupervisedGraphSage_initNetwork
  (JNIEnv *env, jobject obj, jint jinput_dim, jint jnum_class, jintArray joutput_dims) {

  jboolean is_copy;
  auto size = static_cast<size_t>(env->GetArrayLength(joutput_dims));
  int* coutput_dims = static_cast<int*>(env->GetPrimitiveArrayCritical(joutput_dims, &is_copy));

  std::vector<int32_t> output_dims;
  output_dims.resize(size);
  for (size_t i = 0; i < size; i ++)
    output_dims[i] = coutput_dims[i];

  auto* ptr = new angel::graph::SupervisedGraphSage(jinput_dim, jnum_class, output_dims);

  env->ReleasePrimitiveArrayCritical(joutput_dims, coutput_dims, 0);
  return reinterpret_cast<int64_t>(ptr);
}

