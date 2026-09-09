#ifndef TNIE_NN_H
#define TNIE_NN_H

/**
 * @file tnie_nn.h
 * @brief Core data structures and public API for the Tiny Neural Inference Engine.
 *
 * This header defines:
 *  - DenseLayer: a fully-connected neural network layer
 *  - ActivationType: supported activation functions
 *  - TNIE_NeuralNetwork: a simple feed-forward network
 *  - Inference and cleanup functions
 */

#include <stddef.h>  // for size_t

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Supported activation types for each layer.
 */
typedef enum {
    TNIE_ACT_NONE = 0,
    TNIE_ACT_SIGMOID,
    TNIE_ACT_RELU
} TNIE_ActivationType;

/**
 * @brief Status codes returned by TNIE operations.
 */
typedef enum {
    TNIE_OK = 0,
    TNIE_ERROR_INVALID_ARGUMENT = -1,
    TNIE_ERROR_INVALID_NETWORK = -2,
    TNIE_ERROR_INVALID_DIMENSION = -3,
    TNIE_ERROR_INVALID_ACTIVATION = -4,
    TNIE_ERROR_BUFFER_TOO_SMALL = -5,
    TNIE_ERROR_ALLOCATION_FAILED = -6
} TNIE_Status;

/**
 * @brief A simple fully-connected (dense) layer.
 *
 * The weights are stored in row-major order:
 *   - rows   = output_size
 *   - cols   = input_size
 *   - w[o * input_size + i] = weight from input neuron i to output neuron o
 */
typedef struct {
    int   input_size;   /**< Number of input neurons.  */
    int   output_size;  /**< Number of output neurons. */
    float *weights;     /**< Weights array (size = input_size * output_size). */
    float *biases;      /**< Bias array (size = output_size).               */
} TNIE_DenseLayer;

/**
 * @brief A minimal feed-forward neural network.
 *
 * The network is a simple stack of dense layers with an activation
 * function associated to each layer.
 */
typedef struct {
    int                 num_layers;   /**< Number of layers in the network.      */
    TNIE_DenseLayer    *layers;       /**< Array of dense layers.                */
    TNIE_ActivationType *activations; /**< Activation type for each layer.       */
} TNIE_NeuralNetwork;

/**
 * @brief Run a forward pass through the network.
 *
 * @param nn       Pointer to a valid TNIE_NeuralNetwork instance.
 * @param input    Pointer to the input vector of size = first layer input_size.
 * @param output   Pointer to a buffer where the final output will be stored.
 *                 The buffer must have space for at least last_layer->output_size floats.
 *
 * @return 0 on success, non-zero on error.
 *
 * This function allocates temporary working buffers internally and frees
 * them before returning.
 */
int tnie_nn_forward(const TNIE_NeuralNetwork *nn,
                    const float *input,
                    float *output);

/**
 * @brief Run a checked forward pass through the network.
 *
 * Unlike tnie_nn_forward(), this variant verifies the caller-provided buffer
 * sizes before running inference.
 *
 * @param nn           Pointer to a valid neural network.
 * @param input        Input vector.
 * @param input_size   Number of elements available in input.
 * @param output       Output vector.
 * @param output_size  Number of elements available in output.
 *
 * @return TNIE_OK on success, otherwise a TNIE_Status error code.
 */
int tnie_nn_forward_checked(const TNIE_NeuralNetwork *nn,
                            const float *input,
                            size_t input_size,
                            float *output,
                            size_t output_size);

/**
 * @brief Validate a network definition without running inference.
 *
 * @return TNIE_OK when the network is valid, otherwise a TNIE_Status code.
 */
int tnie_nn_validate(const TNIE_NeuralNetwork *nn);

/**
 * @brief Return a stable, human-readable description of a status code.
 */
const char *tnie_status_string(int status);

/**
 * @brief Release all memory associated with a TNIE_NeuralNetwork instance.
 *
 * This function frees:
 *  - weights and biases of each layer
 *  - the layers array
 *  - the activations array
 *  - the TNIE_NeuralNetwork struct itself
 *
 * @param nn  Pointer to a TNIE_NeuralNetwork created on the heap.
 */
void tnie_nn_free(TNIE_NeuralNetwork *nn);

/**
 * @brief Compute the forward pass for a single dense layer.
 *
 * @param layer  Pointer to a valid dense layer.
 * @param input  Input vector (size = layer->input_size).
 * @param output Output buffer (size = layer->output_size).
 * @return TNIE_OK on success, otherwise a TNIE_Status error code.
 */
int tnie_dense_forward(const TNIE_DenseLayer *layer,
                       const float *input,
                       float *output);

#ifdef __cplusplus
}
#endif

#endif // TNIE_NN_H
