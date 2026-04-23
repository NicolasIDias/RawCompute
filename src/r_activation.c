#include <rc/r_types.h>
#include <rc/r_matrix.h>

RMatrix *r_actvation_relu(RNONNULL RMatrix *matrix)
{
    RMatrix *result = r_create_matrix(matrix->rows, matrix->cols);
    for (size_t i = 0; i < matrix->rows; i++)
    {
        for (size_t j = 0; j < matrix->cols; j++)
        {
            float curr = matrix->data[RMatrixIDX(i, j, matrix->cols)];
            int currIndex = RMatrixIDX(i, j, matrix->cols);
            result->data[currIndex]; 
            result->data[currIndex] = (curr > 0) ? curr : 0;
        }
    }
}