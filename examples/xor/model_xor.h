#ifndef TNIE_EXAMPLE_MODEL_XOR_H
#define TNIE_EXAMPLE_MODEL_XOR_H

#include "tnie_nn.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Create the small, hardcoded network used by the XOR example. */
TNIE_NeuralNetwork *tnie_create_xor_demo_network(void);

#ifdef __cplusplus
}
#endif

#endif
