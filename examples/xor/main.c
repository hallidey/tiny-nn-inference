#include <stdio.h>

#include "model_xor.h"
#include "tnie_nn.h"

int main(void) {
    static const float inputs[4][2] = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };
    static const int expected[4] = {0, 1, 1, 0};
    TNIE_NeuralNetwork *nn = tnie_create_xor_demo_network();

    if (!nn) {
        fprintf(stderr, "[TNIE] Failed to create the XOR demo network.\n");
        return 1;
    }

    printf("TNIE - Tiny Neural Inference Engine\n");
    printf("XOR demo with a hardcoded neural network model\n");
    printf("-------------------------------------------------\n");

    for (int i = 0; i < 4; ++i) {
        float output = 0.0f;
        int status = tnie_nn_forward_checked(nn, inputs[i], 2, &output, 1);

        if (status != TNIE_OK) {
            fprintf(stderr, "[TNIE] Sample %d failed: %s (%d)\n",
                    i, tnie_status_string(status), status);
            tnie_nn_free(nn);
            return 1;
        }

        printf("Input: (%.1f, %.1f) -> raw = %.4f, predicted = %d, expected = %d\n",
               inputs[i][0], inputs[i][1], output,
               output > 0.5f ? 1 : 0, expected[i]);
    }

    tnie_nn_free(nn);
    return 0;
}
