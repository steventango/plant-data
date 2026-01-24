import jax.numpy as jnp
import jax.random as jr
import gpjax as gpx


def test_svgp():
    N = 100
    M = 10
    X = jnp.linspace(0, 1, N).reshape(-1, 1)
    y = jnp.sin(X * 6) + jr.normal(jr.PRNGKey(0), (N, 1)) * 0.1
    dataset = gpx.Dataset(X, y)

    z = jnp.linspace(0, 1, M).reshape(-1, 1)

    kernel = gpx.kernels.Matern32()
    mean_function = gpx.mean_functions.Zero()

    # Prior
    prior = gpx.gps.Prior(mean_function=mean_function, kernel=kernel)
    print("Prior initialized.")

    # Likelihood
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=N)
    print("Likelihood initialized.")

    # Variational Family
    q = gpx.variational_families.VariationalGaussian(prior=prior, inducing_inputs=z)
    print("Variational Family initialized.")

    # In 0.13, you might combine them into a Posterior or use directly with SVGP
    # Let's see if Prior * Likelihood works
    try:
        posterior = prior * likelihood
        print("Posterior initialized via multiplication.")
    except Exception as e:
        print(f"Multiplication error: {e}")

    # Check for SVGP model in gpx (might be under gps or a separate module)
    # The 0.13 API uses StochasticVI?
    # Let's check gpx attributes for optimization
    print(f"GPJax features: {dir(gpx)}")


if __name__ == "__main__":
    test_svgp()
