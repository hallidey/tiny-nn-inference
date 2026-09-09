#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "tnie_nn.h"
#include "tnie_activations.h"

int tnie_dense_forward(const TNIE_DenseLayer *layer,
                       const float *input,
                       float *output) {
    if (!layer || !input || !output || !layer->weights || !layer->biases) {
        return TNIE_ERROR_INVALID_ARGUMENT;
    }

    if (layer->input_size <= 0 || layer->output_size <= 0) {
        return TNIE_ERROR_INVALID_DIMENSION;
    }

    int in_size  = layer->input_size;
    int out_size = layer->output_size;

    for (int o = 0; o < out_size; ++o) {
        // Start from the bias term for this output neuron.
        float sum = layer->biases[o];

        // Pointer to the row of weights associated with this output neuron.
        const float *w_row = &layer->weights[o * in_size];

        // Weighted sum of all input neurons.
        for (int i = 0; i < in_size; ++i) {
            sum += w_row[i] * input[i];
        }

        output[o] = sum;
    }

    return TNIE_OK;
}

int tnie_nn_validate(const TNIE_NeuralNetwork *nn) {
    if (!nn || nn->num_layers <= 0 || !nn->layers || !nn->activations) {
        return TNIE_ERROR_INVALID_NETWORK;
    }

    for (int l = 0; l < nn->num_layers; ++l) {
        const TNIE_DenseLayer *layer = &nn->layers[l];

        if (layer->input_size <= 0 || layer->output_size <= 0) {
            return TNIE_ERROR_INVALID_DIMENSION;
        }

        if (!layer->weights || !layer->biases) {
            return TNIE_ERROR_INVALID_NETWORK;
        }

        if (nn->activations[l] < TNIE_ACT_NONE ||
            nn->activations[l] > TNIE_ACT_RELU) {
            return TNIE_ERROR_INVALID_ACTIVATION;
        }

        if (l > 0 && nn->layers[l - 1].output_size != layer->input_size) {
            return TNIE_ERROR_INVALID_DIMENSION;
        }
    }

    return TNIE_OK;
}

static int tnie_nn_forward_impl(const TNIE_NeuralNetwork *nn,
                                const float *input,
                                float *output) {
    int status = tnie_nn_validate(nn);
    if (status != TNIE_OK) {
        return status;
    }

    if (!input || !output) {
        return TNIE_ERROR_INVALID_ARGUMENT;
    }

    // Both alternating buffers must also be able to hold the original input.
    int max_size = nn->layers[0].input_size;
    for (int l = 0; l < nn->num_layers; ++l) {
        if (nn->layers[l].output_size > max_size) {
            max_size = nn->layers[l].output_size;
        }
    }

    if ((size_t)max_size > SIZE_MAX / sizeof(float)) {
        return TNIE_ERROR_INVALID_DIMENSION;
    }

    // Allocate two working buffers to alternate between layers.
    float *buf_a = (float *)malloc(sizeof(float) * max_size);
    float *buf_b = (float *)malloc(sizeof(float) * max_size);
    if (!buf_a || !buf_b) {
        free(buf_a);
        free(buf_b);
        return TNIE_ERROR_ALLOCATION_FAILED;
    }

    // Copy the input into the first buffer.
    const TNIE_DenseLayer *first_layer = &nn->layers[0];
    memcpy(buf_a, input, sizeof(float) * first_layer->input_size);

    float *current_in  = buf_a;
    float *current_out = buf_b;

    // Forward pass through each layer.
    for (int l = 0; l < nn->num_layers; ++l) {
        const TNIE_DenseLayer *layer = &nn->layers[l];

        // Compute linear transformation: y = W * x + b
        status = tnie_dense_forward(layer, current_in, current_out);
        if (status != TNIE_OK) {
            free(buf_a);
            free(buf_b);
            return status;
        }

        // Apply activation in-place on the output buffer.
        status = tnie_apply_activation(nn->activations[l],
                                       current_out,
                                       layer->output_size);
        if (status != TNIE_OK) {
            free(buf_a);
            free(buf_b);
            return status;
        }

        // If this is not the last layer, swap input/output buffers.
        // Otherwise, copy final result into the user-provided output.
        if (l < nn->num_layers - 1) {
            float *tmp   = current_in;
            current_in   = current_out;
            current_out  = tmp;
        } else {
            memcpy(output, current_out,
                   sizeof(float) * layer->output_size);
        }
    }

    free(buf_a);
    free(buf_b);
    return TNIE_OK;
}

int tnie_nn_forward(const TNIE_NeuralNetwork *nn,
                    const float *input,
                    float *output) {
    return tnie_nn_forward_impl(nn, input, output);
}

int tnie_nn_forward_checked(const TNIE_NeuralNetwork *nn,
                            const float *input,
                            size_t input_size,
                            float *output,
                            size_t output_size) {
    int status = tnie_nn_validate(nn);
    if (status != TNIE_OK) {
        return status;
    }

    if (!input || !output) {
        return TNIE_ERROR_INVALID_ARGUMENT;
    }

    if (input_size < (size_t)nn->layers[0].input_size ||
        output_size < (size_t)nn->layers[nn->num_layers - 1].output_size) {
        return TNIE_ERROR_BUFFER_TOO_SMALL;
    }

    return tnie_nn_forward_impl(nn, input, output);
}

const char *tnie_status_string(int status) {
    switch (status) {
        case TNIE_OK:
            return "success";
        case TNIE_ERROR_INVALID_ARGUMENT:
            return "invalid argument";
        case TNIE_ERROR_INVALID_NETWORK:
            return "invalid network";
        case TNIE_ERROR_INVALID_DIMENSION:
            return "invalid dimension";
        case TNIE_ERROR_INVALID_ACTIVATION:
            return "invalid activation";
        case TNIE_ERROR_BUFFER_TOO_SMALL:
            return "buffer too small";
        case TNIE_ERROR_ALLOCATION_FAILED:
            return "allocation failed";
        default:
            return "unknown error";
    }
}

void tnie_nn_free(TNIE_NeuralNetwork *nn) {
    if (!nn) {
        return;
    }

    if (nn->layers) {
        for (int l = 0; l < nn->num_layers; ++l) {
            free(nn->layers[l].weights);
            free(nn->layers[l].biases);
            nn->layers[l].weights = NULL;
            nn->layers[l].biases  = NULL;
        }
        free(nn->layers);
        nn->layers = NULL;
    }

    if (nn->activations) {
        free(nn->activations);
        nn->activations = NULL;
    }

    free(nn);
}
