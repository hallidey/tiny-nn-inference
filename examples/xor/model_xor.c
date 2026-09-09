#include <stdlib.h>

#include "model_xor.h"

static float *allocate_floats(size_t count) {
    return (float *)malloc(sizeof(float) * count);
}

TNIE_NeuralNetwork *tnie_create_xor_demo_network(void) {
    TNIE_NeuralNetwork *nn = (TNIE_NeuralNetwork *)malloc(sizeof(*nn));
    if (!nn) {
        return NULL;
    }

    nn->num_layers = 2;
    nn->layers = (TNIE_DenseLayer *)calloc((size_t)nn->num_layers,
                                           sizeof(*nn->layers));
    nn->activations = (TNIE_ActivationType *)calloc(
        (size_t)nn->num_layers, sizeof(*nn->activations));

    if (!nn->layers || !nn->activations) {
        tnie_nn_free(nn);
        return NULL;
    }

    TNIE_DenseLayer *hidden = &nn->layers[0];
    hidden->input_size = 2;
    hidden->output_size = 2;
    hidden->weights = allocate_floats(4);
    hidden->biases = allocate_floats(2);

    TNIE_DenseLayer *output = &nn->layers[1];
    output->input_size = 2;
    output->output_size = 1;
    output->weights = allocate_floats(2);
    output->biases = allocate_floats(1);

    if (!hidden->weights || !hidden->biases ||
        !output->weights || !output->biases) {
        tnie_nn_free(nn);
        return NULL;
    }

    /* hidden[0] = sigmoid( 20*x0 + 20*x1 - 10) */
    /* hidden[1] = sigmoid(-20*x0 - 20*x1 + 30) */
    hidden->weights[0] = 20.0f;
    hidden->weights[1] = 20.0f;
    hidden->weights[2] = -20.0f;
    hidden->weights[3] = -20.0f;
    hidden->biases[0] = -10.0f;
    hidden->biases[1] = 30.0f;

    /* y = sigmoid(20*hidden[0] + 20*hidden[1] - 30) */
    output->weights[0] = 20.0f;
    output->weights[1] = 20.0f;
    output->biases[0] = -30.0f;

    nn->activations[0] = TNIE_ACT_SIGMOID;
    nn->activations[1] = TNIE_ACT_SIGMOID;

    return nn;
}
