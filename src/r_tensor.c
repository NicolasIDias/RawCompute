#include <rtensor/r_tensor.h>
#include <rtensor/r_types.h>

float r_tensor_dot(RNONNULL RTensor *tensorA, RNONNULL RTensor *tensorB)
{
    if (tensorA->cols != tensorB->cols || tensorA->rows != tensorB->rows)
    {
        printf("[ERROR]: TensorA and TensorB should be the same size");
        exit(1);
    }
    float result = 0.0f;
    size_t total = tensorA->rows * tensorA->cols;
    for (size_t i = 0; i < total; i++)
    {
        result += tensorA->data[i] * tensorB->data[i];
    }
    return result;
}

void r_tensor_matmul(const RNONNULL RTensor *originA, const RNONNULL RTensor *originB, RNONNULL RTensor *dest)
{

    if (originA->cols != originB->rows)
    {
        printf("[ERROR]: originA->cols must equal originB->rows\n");
        exit(1);
    }

    RTensor temp;
    temp.rows = originA->rows;
    temp.cols = originB->cols;
    temp.stride = temp.rows;
    temp.data = malloc(temp.rows * temp.cols * sizeof(float));

    for (int i = 0; i < originA->rows; i++)
        for (int j = 0; j < originB->cols; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < originA->cols; k++)
                sum += originA->data[RTensorIDX(originA, i, k)] * originB->data[RTensorIDX(originB, k, j)];
            temp.data[RTensorIDX(&temp, i, j)] = sum;
        }
    if (dest->data != NULL)
        free(dest->data);
    *dest = temp;
}