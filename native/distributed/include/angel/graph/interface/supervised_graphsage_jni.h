/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class com_tencent_angel_graph_model_SupervisedGraphSage */

#ifndef _Included_com_tencent_angel_graph_model_SupervisedGraphSage
#define _Included_com_tencent_angel_graph_model_SupervisedGraphSage
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     com_tencent_angel_graph_model_SupervisedGraphSage
 * Method:    backward
 * Signature: (J[F[F[II[I[I[I[Ljava/lang/String;)[[F
 */
JNIEXPORT jobjectArray JNICALL Java_com_tencent_angel_graph_model_SupervisedGraphSage_backward
  (JNIEnv *, jobject, jlong, jfloatArray, jintArray, jint, jintArray, jintArray, jintArray, jobjectArray);

/*
 * Class:     com_tencent_angel_graph_model_SupervisedGraphSage
 * Method:    initNetwork
 * Signature: (II[I)J
 */
JNIEXPORT jlong JNICALL Java_com_tencent_angel_graph_model_SupervisedGraphSage_initNetwork
  (JNIEnv *, jobject, jint, jint, jintArray);

#ifdef __cplusplus
}
#endif
#endif
