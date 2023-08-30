# Gaussian Process Regression

## Prediction

Given $N$ training inputs $\mathbf{X}\in\mathbb{R}^{N\times{D}}$ and training targets $\mathbf{y}\in\mathbb{R}^{N}$, the posterior predictive distribution has a closed-form expression:

$$
\begin{aligned}
    p(f_{\star}\rvert\mathbf{y})&=\mathcal{N}(f_{\star}\rvert\mu,\nu),\\
    \mu&=\mathbf{k}_{\star}^{\top}\mathbf{K}_{y}^{-1}\mathbf{y},\label{eq:pred_mu}\\
    \nu&=k_{\star\star}-\mathbf{k}_{\star}^{\top}\mathbf{K}_{y}^{-1}\mathbf{k}_{\star},
\end{aligned}
$$

where $\mathbf{k}_{\star}$ is the vector of kernel values between all the training inputs $\mathbf{X}$ and the test input $\mathbf{x}^{\star}$, $\mathbf{K}_{y}$ is a shorthand of $\mathbf{K}_{\mathbf{x}}+\sigma^{2}\mathbf{I}$, $\mathbf{K}_{\mathbf{x}}$ is the covariance matrix given by the kernel function evaluated at each pair of training inputs, and $k_{\star\star}\triangleq\mathtt{k}(\mathbf{x}^{\star},\mathbf{x}^{\star})$.

## Learning

Optimizing the hyperparameters -- a process known as model selection -- is a common practice to obtain a better prediction.
Model selection is typically implemented by maximizing the model evidence (better known as log marginal likelihood)

$$
\ln{p(\mathbf{y}|\mathbf{\psi})}=\frac{1}{2}(\underbrace{-\mathbf{y}^{\top}\mathbf{K}_{y}^{-1}\mathbf{y}}_{\text{quadratic term}}-\underbrace{\ln{\mathrm{det}(\mathbf{K}_{y})}}_{\text{logdet term}}-\underbrace{N\ln(2\pi)}_{\text{constant term}}),
$$
