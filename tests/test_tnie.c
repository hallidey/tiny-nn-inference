#include <math.h>
#include <stdio.h>

#include "tnie_activations.h"
#include "model_xor.h"
#include "tnie_nn.h"

static int failures = 0;

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,           \
                    #condition);                                                \
            ++failures;                                                         \
        }                                                                       \
    } while (0)

static int nearly_equal(float actual, float expected, float tolerance) {
    return fabsf(actual - expected) <= tolerance;
}

static void test_activations(void) {
    float values[] = {-2.0f, 0.0f, 3.0f};

    CHECK(tnie_apply_activation(TNIE_ACT_RELU, values, 3) == TNIE_OK);
    CHECK(nearly_equal(values[0], 0.0f, 0.0001f));
    CHECK(nearly_equal(values[1], 0.0f, 0.0001f));
    CHECK(nearly_equal(values[2], 3.0f, 0.0001f));
    CHECK(nearly_equal(tnie_sigmoidf(0.0f), 0.5f, 0.0001f));
    CHECK(tnie_apply_activation((TNIE_ActivationType)99, values, 3) ==
          TNIE_ERROR_INVALID_ACTIVATION);
}

static void test_dense_layer(void) {
    float weights[] = {1.0f, 2.0f, -1.0f, 1.0f};
    float biases[] = {0.5f, -0.5f};
    float input[] = {2.0f, 3.0f};
    float output[2] = {0.0f, 0.0f};
    TNIE_DenseLayer layer = {2, 2, weights, biases};

    CHECK(tnie_dense_forward(&layer, input, output) == TNIE_OK);
    CHECK(nearly_equal(output[0], 8.5f, 0.0001f));
    CHECK(nearly_equal(output[1], 0.5f, 0.0001f));
}

static void test_xor_model(void) {
    const float inputs[4][2] = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };
    const int expected[4] = {0, 1, 1, 0};
    TNIE_NeuralNetwork *nn = tnie_create_xor_demo_network();

    CHECK(nn != NULL);
    if (!nn) {
        return;
    }

    CHECK(tnie_nn_validate(nn) == TNIE_OK);
    for (int i = 0; i < 4; ++i) {
        float output = 0.0f;
        CHECK(tnie_nn_forward_checked(nn, inputs[i], 2, &output, 1) == TNIE_OK);
        CHECK((output > 0.5f ? 1 : 0) == expected[i]);
    }

    {
        float input[] = {0.0f, 1.0f};
        float output = 0.0f;
        CHECK(tnie_nn_forward_checked(nn, input, 1, &output, 1) ==
              TNIE_ERROR_BUFFER_TOO_SMALL);
        CHECK(tnie_nn_forward_checked(nn, input, 2, &output, 0) ==
              TNIE_ERROR_BUFFER_TOO_SMALL);
    }

    tnie_nn_free(nn);
}

static void test_dimension_validation(void) {
    float first_weights[4] = {0};
    float first_biases[2] = {0};
    float second_weights[3] = {0};
    float second_biases[1] = {0};
    TNIE_DenseLayer layers[] = {
        {2, 2, first_weights, first_biases},
        {3, 1, second_weights, second_biases}
    };
    TNIE_ActivationType activations[] = {TNIE_ACT_RELU, TNIE_ACT_NONE};
    TNIE_NeuralNetwork nn = {2, layers, activations};

    CHECK(tnie_nn_validate(&nn) == TNIE_ERROR_INVALID_DIMENSION);
}

static void test_input_larger_than_output(void) {
    float weights[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float biases[] = {0.0f};
    float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output = 0.0f;
    TNIE_DenseLayer layer = {4, 1, weights, biases};
    TNIE_ActivationType activation = TNIE_ACT_NONE;
    TNIE_NeuralNetwork nn = {1, &layer, &activation};

    CHECK(tnie_nn_forward_checked(&nn, input, 4, &output, 1) == TNIE_OK);
    CHECK(nearly_equal(output, 10.0f, 0.0001f));
}

int main(void) {
    test_activations();
    test_dense_layer();
    test_xor_model();
    test_dimension_validation();
    test_input_larger_than_output();

    if (failures != 0) {
        fprintf(stderr, "%d test(s) failed\n", failures);
        return 1;
    }

    printf("All TNIE tests passed.\n");
    return 0;
}
