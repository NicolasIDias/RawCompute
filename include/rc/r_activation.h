#ifndef RC_R_ACTIVATION_H
#define RC_R_ACTIVATION_H

#include <rc/r_types.h>
#include <rc/r_matrix.h>

RMatrix *r_activation_relu(RNONNULL RMatrix *matrix);
RMatrix *r_activation_leaky_relu(RNONNULL RMatrix *matrix, float alpha);
RMatrix *r_activation_gelu(RNONNULL RMatrix *matrix);
RMatrix *r_activation_softmax(RNONNULL RMatrix *matrix);
RMatrix *r_activation_swish(RNONNULL RMatrix *matrix);

#endif
