//
// Created by leleyu on 2019-05-05.
//

#include <angel/pytorch/lr_jni.h>
#include <angel/pytorch/lr.h>

#include <angel/commons.h>

/*
 * Class:     com_tencent_angel_pytorch_model_LogisticRegression
 * Method:    nativeInitPtr
 * Signature: (Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_com_tencent_angel_pytorch_model_LogisticRegression_nativeInitPtr
  (JNIEnv *env, jobject jobj, jstring jpath) {
  DEFINE_STRING(jpath);
  auto* ptr = new angel::LogisticRegression(jpath_cstr);
  RELEASE_STRING(jpath);
  return reinterpret_cast<int64_t>(ptr);
}

/*
 * Class:     com_tencent_angel_pytorch_model_LogisticRegression
 * Method:    nativeDestroyPtr
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_tencent_angel_pytorch_model_LogisticRegression_nativeDestroyPtr
  (JNIEnv *env, jobject jobj, jlong jptr) {
  DEFINE_MODEL_PTR(angel::LogisticRegression, jptr);
  delete(ptr);
}

/*
 * Class:     com_tencent_angel_pytorch_model_LogisticRegression
 * Method:    nativeForward
 * Signature: (JI[I[I[IF[F)[F
 */
JNIEXPORT jfloatArray JNICALL Java_com_tencent_angel_pytorch_model_LogisticRegression_nativeForward
  (JNIEnv *env, jobject jobj, jlong jptr, jint jbatch_size, jlongArray jindex,
    jlongArray jfeats, jfloatArray jvalues, jfloatArray jbias, jfloatArray jweights) {
  jboolean is_copy;
  DEFINE_MODEL_PTR(angel::LogisticRegression, jptr);
  DEFINE_PRIMITIVE_ARRAYS5(jindex, jfeats, jvalues, jbias, jweights);

  DEFINE_TORCH_TENSOR_ARRAY(jindex, torch::kInt64)
  DEFINE_TORCH_TENSOR_ARRAY(jfeats, torch::kInt64)
  DEFINE_TORCH_TENSOR_ARRAY(jvalues, torch::kF32)
  DEFINE_TORCH_TENSOR_ARRAY(jbias, torch::kF32)
  DEFINE_TORCH_TENSOR_ARRAY(jweights, torch::kF32)
  DEFINE_TORCH_TENSOR_SCALA(jbatch_size, torch::kInt32)

  std::vector<torch::jit::IValue> inputs;
  inputs.resize(6);
  inputs[0] = jbatch_size_tensor;
  inputs[1] = jindex_tensor;
  inputs[2] = jfeats_tensor;
  inputs[3] = jvalues_tensor;
  inputs[4] = jbias_tensor;
  inputs[5] = jweights_tensor;

  auto output = ptr->forward(inputs);
  auto output_ptr = output.data_ptr();
  DEFINE_JFLOATARRAY(output_ptr, jbatch_size);
  RELEASE_PRIMITIVE_ARRAYS5(jindex, jfeats, jvalues, jbias, jweights);
  return output_ptr_jarray;
}

/*
 * Class:     com_tencent_angel_pytorch_model_LogisticRegression
 * Method:    nativeBackward
 * Signature: (JI[I[I[IF[F[F)V
 */
JNIEXPORT void JNICALL Java_com_tencent_angel_pytorch_model_LogisticRegression_nativeBackward
  (JNIEnv *env, jobject jobj, jlong jptr, jint jbatch_size, jlongArray jindex,
    jlongArray jfeats, jfloatArray jvalues, jfloatArray jbias, jfloatArray jweights,
    jfloatArray jtargets) {
  jboolean is_copy;
  DEFINE_MODEL_PTR(angel::LogisticRegression, jptr);
  DEFINE_PRIMITIVE_ARRAYS6(jindex, jfeats, jvalues, jweights, jbias, jtargets);

  DEFINE_TORCH_TENSOR_ARRAY(jindex, torch::kInt64)
  DEFINE_TORCH_TENSOR_ARRAY(jfeats, torch::kInt64)
  DEFINE_TORCH_TENSOR_ARRAY(jvalues, torch::kF32)
  DEFINE_TORCH_TENSOR_ARRAY_GRAD(jbias, torch::kF32)
  DEFINE_TORCH_TENSOR_ARRAY_GRAD(jweights, torch::kF32)
  DEFINE_TORCH_TENSOR_ARRAY(jtargets, torch::kF32)
  DEFINE_TORCH_TENSOR_SCALA(jbatch_size, torch::kInt32)

  ptr->backward(jbatch_size_tensor, jindex_tensor, jfeats_tensor,
    jvalues_tensor, jbias_tensor, jweights_tensor, jtargets_tensor);

  // set the grad and return
  env->SetFloatArrayRegion(jbias, 0, env->GetArrayLength(jbias), jbias_tensor.grad().data<float>());
  env->SetFloatArrayRegion(jweights, 0, env->GetArrayLength(jweights), jweights_tensor.grad().data<float>());

  RELEASE_PRIMITIVE_ARRAYS6(jindex, jfeats, jvalues, jbias, jweights, jtargets);
}
