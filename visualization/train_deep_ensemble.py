import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax as ox
import polars as pl
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler

# 1. Architecture: Heteroscedastic MLP
class HeteroscedasticMLP(eqx.Module):
    layers: list

    def __init__(self, in_size, out_size, key):
        keys = jr.split(key, 3)
        self.layers = [
            eqx.nn.Linear(in_size, 128, key=keys[0]),
            jax.nn.relu,
            eqx.nn.Linear(128, 128, key=keys[1]),
            jax.nn.relu,
            eqx.nn.Linear(128, out_size * 2, key=keys[2]) # [mean, log_var]
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        # Output [mean, log_var]
        mean, log_var = jnp.split(x, 2)
        log_var = jnp.clip(log_var, -8, 2)
        return mean, log_var

# 2. Vectorized Training Logic
@eqx.filter_jit
def nll_loss(model, x, y):
    # Vectorize prediction over batch
    means, log_vars = jax.vmap(model)(x)
    prec = jnp.exp(-log_vars)
    diff = jnp.square(y - means)
    return jnp.mean(0.5 * (prec * diff + log_vars))

@eqx.filter_jit
def ensemble_train_step(stacked_arrays, static, opt_states, x, y, optim):
    def step(arr, s):
        m = eqx.combine(arr, static)
        loss_val, grads = eqx.filter_value_and_grad(nll_loss)(m, x, y)
        updates, s = optim.update(grads, s, eqx.filter(m, eqx.is_array))
        m = eqx.apply_updates(m, updates)
        new_arr, _ = eqx.partition(m, eqx.is_array)
        return new_arr, s, loss_val

    # vmap over the ensemble dimension
    stacked_arrays, opt_states, losses = jax.vmap(step)(stacked_arrays, opt_states)
    return stacked_arrays, opt_states, jnp.mean(losses)

def main():
    # 3. Data Loading
    path = "/data/plant-rl/offline/v22/mixed-v22.parquet"
    print(f"Loading full dataset from {path}...")
    df_raw = pl.read_parquet(path)
    
    # Dynamics with fixed Polars logic
    df = df_raw.with_columns(
        (pl.col("time") - pl.col("time").min().over("experiment", "zone", "plant_id")).dt.total_seconds() / (24 * 3600)
    ).rename({"time":"days"}) 
    # Use modern get/alias for list expansion
    df = df.with_columns([
        pl.col("cls_token_pca").list.get(i).alias(f"pca_{i}") for i in range(5)
    ]).filter((pl.col("days") <= 14) & pl.col("valid") & (pl.col("clean_area") > 0))
    
    action_cols = ["red_coef", "white_coef", "blue_coef"]
    feature_cols = ["clean_area"] + action_cols + [f"pca_{i}" for i in range(5)]
    
    df_dyn = df.with_columns([
        (pl.col("clean_area").log().shift(-1).over("experiment", "zone", "plant_id") - pl.col("clean_area").log()).alias("delta_ln_area"),
    ]).filter(pl.col("delta_ln_area").is_not_null() & pl.col("delta_ln_area").is_finite())
    
    X_raw = df_dyn[feature_cols].to_numpy()
    X_raw[:, 0] = np.log(X_raw[:, 0]) # ln(Area)
    Y_raw = df_dyn["delta_ln_area"].to_numpy().reshape(-1, 1)
    
    scaler = StandardScaler()
    X_scaled = jnp.array(scaler.fit_transform(X_raw))
    Y = jnp.array(Y_raw)
    
    # 4. Correct Ensemble Parallelization
    ensemble_size = 5
    print(f"\nInitializing ensemble of {ensemble_size} members...")
    key = jr.PRNGKey(42)
    
    arrays_list = []
    static_ref = None
    for i in range(ensemble_size):
        m_key, key = jr.split(key)
        m = HeteroscedasticMLP(X_scaled.shape[1], 1, m_key)
        arr, static = eqx.partition(m, eqx.is_array)
        arrays_list.append(arr)
        if static_ref is None: static_ref = static
    
    stacked_arrays = jax.tree.map(lambda *args: jnp.stack(args), *arrays_list)
    optim = ox.adam(1e-3)
    opt_states = jax.vmap(optim.init)(stacked_arrays)

    # 5. Training Loop
    epochs = 1500
    batch_size = 1024
    num_batches = len(X_scaled) // batch_size
    print(f"Starting parallel training on {len(X_raw)} samples (Ensemble Size={ensemble_size})...")
    p_key = jr.PRNGKey(0)
    for epoch in range(epochs):
        p_key, subkey = jr.split(p_key)
        indices = jr.permutation(subkey, len(X_scaled))
        epoch_loss = 0
        for b in range(num_batches):
            idx = indices[b * batch_size : (b + 1) * batch_size]
            stacked_arrays, opt_states, loss = ensemble_train_step(
                stacked_arrays, static_ref, opt_states, X_scaled[idx], Y[idx], optim
            )
            epoch_loss += loss
        if epoch % 300 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:4d} | Ensemble Mean NLL: {epoch_loss/num_batches:.4f}")

    # 6. Serialization
    os.makedirs("results/ensemble_models", exist_ok=True)
    for i in range(ensemble_size):
        m_arr = jax.tree.map(lambda x: x[i], stacked_arrays)
        member = eqx.combine(m_arr, static_ref)
        eqx.tree_serialise_leaves(f"results/ensemble_models/member_{i}.eqx", member)
    with open("results/ensemble_models/metadata.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "feature_cols": feature_cols}, f)
    print("\nParallel Deep Ensemble training complete.")

if __name__ == "__main__":
    main()
