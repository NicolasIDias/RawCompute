#include <rc/r_activation_grad.h>

void r_activation_relu_grad(const RNONNULL RMatrix *input, const RNONNULL RMatrix *upstream_grad,
                            RNONNULL RMatrix *grad)
{
    for (size_t i = 0; i < input->rows; i++)
    {
        for (size_t j = 0; j < input->cols; j++)
        {
            size_t index = RMatrixIDX(i, j, input->cols);
            grad->data[index] = (input->data[index] > 0) ? upstream_grad->data[index] : 0;
        }
    }
}

void r_activation_leaky_relu_grad(const RNONNULL RMatrix *input, const RNONNULL RMatrix *upstream_grad, float alpha,
                                  RNONNULL RMatrix *grad)
{
    for (size_t i = 0; i < input->rows; i++)
    {
        for (size_t j = 0; j < input->cols; j++)
        {
            size_t index = RMatrixIDX(i, j, input->cols);
            grad->data[index] =
                (input->data[index] > 0) ? upstream_grad->data[index] : upstream_grad->data[index] * alpha;
        }
    }
}

void r_activation_gelu_grad(const RNONNULL RMatrix *input, const RNONNULL RMatrix *upstream_grad,
                            RNONNULL RMatrix *grad)
{
    float bsqrt2 = 1 / M_SQRT2;
    float inv_sqrt_2pi = 1.0f / sqrtf(2.0f * M_PI);

    for (size_t i = 0; i < input->rows; i++)
    {
        for (size_t j = 0; j < input->cols; j++)
        {
            size_t index = RMatrixIDX(i, j, input->cols);

            float val = input->data[index];

            float cdf = 0.5 * (1 + erff(val * bsqrt2));
            float pdf = inv_sqrt_2pi * expf(-0.5f * val * val);

            float local_grad = cdf + (val * pdf);

            grad->data[index] = local_grad * upstream_grad->data[index];
        }
    }
}

void r_activation_softmax_grad(const RNONNULL RMatrix *output, const RNONNULL RMatrix *upstream_grad,
                               RNONNULL RMatrix *grad)
{
    for (size_t i = 0; i < output->rows; i++)
    {
        float dot = 0.0f;
        for (size_t j = 0; j < output->cols; j++)
        {
            size_t index = RMatrixIDX(i, j, output->cols);
            dot += upstream_grad->data[index] * output->data[index];
        }

        for (size_t j = 0; j < output->cols; j++)
        {
            size_t index = RMatrixIDX(i, j, output->cols);
            grad->data[index] = output->data[index] * (upstream_grad->data[index] - dot);
        }
    }
}

void r_activation_swish_grad(const RNONNULL RMatrix *input, const RNONNULL RMatrix *upstream_grad,
                             RNONNULL RMatrix *grad)
{
    for (size_t i = 0; i < input->rows; i++)
    {
        for (size_t j = 0; j < input->cols; j++)
        {
            size_t index = RMatrixIDX(i, j, input->cols);
            float val = input->data[index];
            float sigmoid = 1.0f / (1.0f + expf(-val));
            float local_grad = sigmoid + val * sigmoid * (1.0f - sigmoid);

            grad->data[index] = upstream_grad->data[index] * local_grad;
        }
    }
}
