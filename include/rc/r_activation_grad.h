#ifndef RC_R_ACTIVATION_GRAD_H
#define RC_R_ACTIVATION_GRAD_H

#include <rc/r_types.h>
#include <rc/r_matrix.h>

void r_activation_relu_grad(const RNONNULL RMatrix *input, const RNONNULL RMatrix *upstream_grad, RNONNULL RMatrix *grad);
void r_activation_leaky_relu_grad(const RNONNULL RMatrix *input, const RNONNULL RMatrix *upstream_grad, float alpha, RNONNULL RMatrix *grad);
void r_activation_gelu_grad(const RNONNULL RMatrix *input, const RNONNULL RMatrix *upstream_grad, RNONNULL RMatrix *grad);
void r_activation_softmax_grad(const RNONNULL RMatrix *output, const RNONNULL RMatrix *upstream_grad, RNONNULL RMatrix *grad);
void r_activation_swish_grad(const RNONNULL RMatrix *input, const RNONNULL RMatrix *upstream_grad, RNONNULL RMatrix *grad);

#endif