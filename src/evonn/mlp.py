"""One-hidden-layer MLP implemented in numpy, shared by ALL methods.

Parameters live in a single flat vector theta (so EAs can treat it as a genome
and scipy can treat it as an optimisation variable).

Planned API:
  - n_params(d_in, hidden, d_out)
  - unpack(theta) -> (W1, b1, W2, b2)
  - forward(theta, X) -> predictions
  - loss(theta, X, y) -> scalar (MSE or cross-entropy)
  - loss_and_grad(theta, X, y) -> (scalar, grad)  # manual backprop
"""
