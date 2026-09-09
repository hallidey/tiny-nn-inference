#ifndef TNIE_ACTIVATIONS_H
#define TNIE_ACTIVATIONS_H

/**
 * @file tnie_activations.h
 * @brief Activation functions used by the Tiny Neural Inference Engine.
 */

#include "tnie_nn.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Sigmoid activation for a single scalar.
 */
float tnie_sigmoidf(float x);

/**
 * @brief ReLU activation for a single scalar.
 */
float tnie_reluf(float x);

/**
 * @brief Apply the given activation function to a vector in-place.
 *
 * @param type  Activation type (NONE, SIGMOID, RELU).
 * @param data  Pointer to the vector to transform.
 * @param size  Number of elements in the vector.
 * @return TNIE_OK on success, otherwise a TNIE_Status error code.
 */
int tnie_apply_activation(TNIE_ActivationType type, float *data, int size);

#ifdef __cplusplus
}
#endif

#endif // TNIE_ACTIVATIONS_H
