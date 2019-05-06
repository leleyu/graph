/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class com_tencent_angel_pytorch_model_LogisticRegression */

#ifndef _Included_com_tencent_angel_pytorch_model_LogisticRegression
#define _Included_com_tencent_angel_pytorch_model_LogisticRegression
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     com_tencent_angel_pytorch_model_LogisticRegression
 * Method:    nativeInitPtr
 * Signature: (Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_com_tencent_angel_pytorch_model_LogisticRegression_nativeInitPtr
  (JNIEnv *, jobject, jstring);

/*
 * Class:     com_tencent_angel_pytorch_model_LogisticRegression
 * Method:    nativeDestroyPtr
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_tencent_angel_pytorch_model_LogisticRegression_nativeDestroyPtr
  (JNIEnv *, jobject, jlong);

/*
 * Class:     com_tencent_angel_pytorch_model_LogisticRegression
 * Method:    nativeForward
 * Signature: (JI[J[J[FF[F)[F
 */
JNIEXPORT jfloatArray JNICALL Java_com_tencent_angel_pytorch_model_LogisticRegression_nativeForward
  (JNIEnv *, jobject, jlong, jint, jlongArray, jlongArray, jfloatArray, jfloatArray , jfloatArray);

/*
 * Class:     com_tencent_angel_pytorch_model_LogisticRegression
 * Method:    nativeBackward
 * Signature: (JI[J[J[FF[F[F)V
 */
JNIEXPORT void JNICALL Java_com_tencent_angel_pytorch_model_LogisticRegression_nativeBackward
  (JNIEnv *, jobject, jlong, jint, jlongArray, jlongArray, jfloatArray, jfloatArray, jfloatArray, jfloatArray);

#ifdef __cplusplus
}
#endif
#endif