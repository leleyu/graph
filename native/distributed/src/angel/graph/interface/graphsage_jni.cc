//
// Created by leleyu on 2019-04-08.
//

#include <angel/graph/interface/graphsage_jni.h>
#include <angel/graph/interface/commons.h>
#include <angel/graph/model/graphsage.h>

/*
 * Class:     com_tencent_angel_graph_model_GraphSage
 * Method:    forward
 * Signature: (J[F[F[I[I[I)[F
 */
JNIEXPORT jfloatArray JNICALL Java_com_tencent_angel_graph_model_GraphSage_forward
  (JNIEnv *env, jobject jobj, jlong jptr,
    jfloatArray jinput_embeddings,
    jintArray jbatch, // batch of node
    jint jmax_neibor, // maximum number of neibors
    jintArray jnodes, // graph structures
    jintArray jneibors) {
  jboolean is_copy;

  // Get primitive arrays.
  DEFINE_PRIMITIVE_ARRAYS4(jinput_embeddings, jbatch, jnodes, jneibors);
  // model ptr
  DEFINE_MODEL_PTR(angel::graph::SupervisedGraphSage, jptr);
  int embedding_dim = ptr->GetDim();
  // embeddings
  DEFINE_EMBEDDINGS(jinput_embeddings, embedding_dim);
  // graph structures
  DEFINE_GRAPH_STRUCTURE(jnodes, jneibors, jmax_neibor);
  // input nodes
  DEFINE_TORCH_TENSOR(jbatch, torch::kInt64);
  // Forward
  auto output = ptr->Forward(jbatch_tensor, sub_graph, input_embeddings);
  // Release them
  RELEASE_PRIMITIVE_ARRAYS4(jinput_embeddings, jbatch, jnodes, jneibors);

  // Return
  auto output_ptr = output.data_ptr();
  DEFINE_JFLOATARRAY(output_ptr, output.size(0));
  return output_ptr_jarray;
}


/*
 * Class:     com_tencent_angel_graph_model_GraphSage
 * Method:    destroyNetwork
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_tencent_angel_graph_model_GraphSage_destroyNetwork
  (JNIEnv * env, jobject jobj, jlong jptr) {
  auto* ptr = reinterpret_cast<angel::graph::SupervisedGraphSage*>(jptr);
  delete(ptr);
}

/*
 * Class:     com_tencent_angel_graph_model_GraphSage
 * Method:    getKeys
 * Signature: (J)[Ljava/lang/String;
 */
JNIEXPORT jobjectArray JNICALL Java_com_tencent_angel_graph_model_GraphSage_getKeys
    (JNIEnv *env, jobject jobj, jlong jptr) {
  auto* ptr = reinterpret_cast<angel::graph::SupervisedGraphSage*>(jptr);
  std::vector<std::string> keys = ptr->keys();

  auto len = static_cast<int>(keys.size());
  jclass stringClass = env->FindClass("java/lang/String");
  jobjectArray strings = env->NewObjectArray(len, stringClass, env->NewStringUTF(""));
  for (jsize i = 0; i < len; i++) {
    env->SetObjectArrayElement(strings, i, env->NewStringUTF(keys[i].data()));
  }

  return strings;
}