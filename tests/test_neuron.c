#include <rtensor/r_types.h>
#include <rtensor/r_layer_dense.h>

int main(void)
{
    RNeuron neuron1 = {.bias = 2.0,
                      .n_inputs = 3};

    neuron1.weights = malloc(sizeof(float) * neuron1.n_inputs);
    neuron1.weights[0] = 0.2f;
    neuron1.weights[1] = 0.8f;
    neuron1.weights[2] = -0.5f;

    float inputs1[] = {1.0f, 2.0f, 3.0f};
    int input_count1 = sizeof(inputs1) / sizeof(inputs1[0]);

    float output1 = r_calc_neuron_output(&neuron1, inputs1, input_count1);

    printf("%f\n", output1);

    free(neuron1.weights);
    neuron1.bias = 0;
    neuron1.n_inputs = 0;
    

    RNeuron neuron2 = {.bias = 2.0,
                      .n_inputs = 4};

    neuron2.weights = malloc(sizeof(float) * neuron2.n_inputs);
    neuron2.weights[0] = 0.2f;
    neuron2.weights[1] = 0.8f;
    neuron2.weights[2] = -0.5f;
    neuron2.weights[3] = 1.0f;

    float inputs2[] = {1.0f, 2.0f, 3.0f, 2.5f};
    int input_count2 = sizeof(inputs2) / sizeof(inputs2[0]);

    float output2 = r_calc_neuron_output(&neuron2, inputs2, input_count2);

    printf("%f\n", output2);

    free(neuron2.weights);
    neuron2.bias = 0;
    neuron2.n_inputs = 0;

    return 0;
}