import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ml.generative_trainer import GenerativeTrainer
from utils.diagnostics import print_distribution_stats, compute_acf # Updated import
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    # ... (Configuration reste identique) ...
    config = {
        'n_steps': 60,
        'T': 0.25,
    }

    print("--- 1. Initializing Generative Quant Engine ---")
    trainer = GenerativeTrainer(config)

    real_data_np = np.array(trainer.market_paths)
    print_distribution_stats("REAL MARKET DATA (TARGET)", real_data_np)

    # Train (150-300 epochs)
    trained_model = trainer.run(n_epochs=200, batch_size=256)

    print("\n--- 2. Generating Synthetic Market Scenarios ---")
    key = jax.random.PRNGKey(777)

    n_gen = 2000
    noise = jax.random.normal(key, (n_gen, config['n_steps']))
    noise_sigs = trainer.sig_extractor.get_signature(noise)

    real_v0s = trainer.market_paths[:, 0]
    random_indices = jax.random.randint(key, (n_gen,), 0, len(real_v0s))
    v0_samples = real_v0s[random_indices]

    dt = config['T'] / config['n_steps']
    fake_vars = jax.vmap(trained_model.generate_variance_path, in_axes=(0, 0, 0, None))(
        v0_samples, noise_sigs, noise, dt
    )
    fake_vars_np = np.array(fake_vars)

    print_distribution_stats("AI GENERATED DATA", fake_vars_np)

    print("\n--- 3. Visualizing Stylized Facts ---")

    # Create 3 subplots: Paths, Density, and Auto-Correlation
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.4, 0.3, 0.3],
        vertical_spacing=0.1,
        subplot_titles=("Volatility Paths", "Density Distribution (Fat Tails)", "Auto-Correlation (Memory)")
    )

    # 1. Paths
    for i in range(50):
        fig.add_trace(go.Scatter(y=real_data_np[i], mode='lines', line=dict(color='rgba(200, 200, 200, 0.3)', width=1), showlegend=False), row=1, col=1)
    for i in range(50):
        fig.add_trace(go.Scatter(y=fake_vars_np[i], mode='lines', line=dict(color='rgba(0, 255, 255, 0.4)', width=1), showlegend=False), row=1, col=1)

    # 2. Histogram
    range_max = 0.15 
    fig.add_trace(go.Histogram(x=real_data_np.flatten(), xbins=dict(start=0, end=range_max, size=0.002), marker_color='white', opacity=0.6, name='Real', histnorm='probability density'), row=2, col=1)
    fig.add_trace(go.Histogram(x=fake_vars_np.flatten(), xbins=dict(start=0, end=range_max, size=0.002), marker_color='cyan', opacity=0.6, name='AI', histnorm='probability density'), row=2, col=1)

    # 3. Auto-Correlation (The Quant Check)
    # Average ACF over 100 paths to reduce noise
    real_acf = np.mean([compute_acf(p, lags=20) for p in real_data_np[:100]], axis=0)
    fake_acf = np.mean([compute_acf(p, lags=20) for p in fake_vars_np[:100]], axis=0)
    lags = np.arange(20)

    fig.add_trace(go.Scatter(x=lags, y=real_acf, mode='lines+markers', name='Real ACF', line=dict(color='white', dash='dash')), row=3, col=1)
    fig.add_trace(go.Scatter(x=lags, y=fake_acf, mode='lines+markers', name='AI ACF', line=dict(color='cyan')), row=3, col=1)

    fig.update_layout(title='Generative Neural SDE: Statistical Validation', template='plotly_dark', height=1000)
    fig.show()

if __name__ == "__main__":
    main()