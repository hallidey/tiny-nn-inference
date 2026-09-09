#include <math.h>
#include "tnie_activations.h"

float tnie_sigmoidf(float x) {
    // Classic sigmoid: 1 / (1 + e^(-x))
    return 1.0f / (1.0f + expf(-x));
}

float tnie_reluf(float x) {
    // Rectified Linear Unit: max(0, x)
    return (x > 0.0f) ? x : 0.0f;
}

int tnie_apply_activation(TNIE_ActivationType type, float *data, int size) {
    if (!data || size <= 0) {
        return TNIE_ERROR_INVALID_ARGUMENT;
    }

    if (type == TNIE_ACT_NONE) {
        return TNIE_OK; // no-op
    }

    if (type != TNIE_ACT_SIGMOID && type != TNIE_ACT_RELU) {
        return TNIE_ERROR_INVALID_ACTIVATION;
    }

    for (int i = 0; i < size; ++i) {
        switch (type) {
            case TNIE_ACT_SIGMOID:
                data[i] = tnie_sigmoidf(data[i]);
                break;
            case TNIE_ACT_RELU:
                data[i] = tnie_reluf(data[i]);
                break;
            case TNIE_ACT_NONE:
            default:
                break;
        }
    }

    return TNIE_OK;
}
