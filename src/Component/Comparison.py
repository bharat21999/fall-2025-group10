
# compare_methods.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("/home/ubuntu/Capstone/results")
#DATA_DIR = Path("/home/ubuntu/Capstone/data/cmab")

def load_results():
    """Load all method results"""
    results = {}

    # LinUCB (with features)
    try:
        results['LinUCB'] = pd.read_parquet(OUT_DIR / "linucb_history.parquet")
        print(f"✓ Loaded LinUCB: {len(results['LinUCB']):,} rounds")
    except FileNotFoundError:
        print("✗ LinUCB results not found. Run main.py first.")
        results['LinUCB'] = None

    # UCB1 (no features)
    try:
        results['UCB1'] = pd.read_parquet(OUT_DIR / "ucb1_history.parquet")
        print(f"✓ Loaded UCB1: {len(results['UCB1']):,} rounds")
    except FileNotFoundError:
        print("✗ UCB1 results not found. Run mab_ucb1.py first.")
        results['UCB1'] = None

    # Random
    try:
        results['Random'] = pd.read_parquet(OUT_DIR / "random_history.parquet")
        print(f"✓ Loaded Random: {len(results['Random']):,} rounds")
    except FileNotFoundError:
        print("✗ Random results not found. Run random_recommender.py first.")
        results['Random'] = None

    # Collaborative Filtering (CF)
    try:
        results['CF'] = pd.read_parquet(OUT_DIR / "cf_history.parquet")
        print(f"✓ Loaded CF: {len(results['CF']):,} rounds")
    except FileNotFoundError:
        print("✗ CF results not found. Run CF_recommender.py first.")
        results['CF'] = None

    return results


def print_comparison_table(results):
    """Print comprehensive comparison table"""
    print("\n" + "=" * 90)
    print("COMPARISON: LinUCB vs UCB1 vs Random vs CF")
    print("=" * 90)

    # Filter out None results
    methods = {k: v for k, v in results.items() if v is not None}

    if not methods:
        print("No results to compare!")
        return

    print(f"\n{'Metric':<35s}", end='')
    for method in methods.keys():
        print(f"{method:>15s}", end='')
    print(f"{'Best':>12s}")
    print("-" * 90)

    # Total rounds
    print(f"{'Total rounds':<35s}", end='')
    for df in methods.values():
        print(f"{len(df):>15,}", end='')
    print()

    # Average regret
    print(f"{'Average regret':<35s}", end='')
    avg_regrets = {}
    for name, df in methods.items():
        avg = df['regret'].mean()
        avg_regrets[name] = avg
        print(f"{avg:>15.6f}", end='')
    best = min(avg_regrets, key=avg_regrets.get)
    print(f"{best + ' ✓':>12s}")

    # Cumulative regret
    print(f"{'Cumulative regret':<35s}", end='')
    cum_regrets = {}
    for name, df in methods.items():
        cum = df['regret'].sum()
        cum_regrets[name] = cum
        print(f"{cum:>15.2f}", end='')
    best = min(cum_regrets, key=cum_regrets.get)
    print(f"{best + ' ✓':>12s}")

    # Standard deviation of regret
    print(f"{'Std dev of regret':<35s}", end='')
    for df in methods.values():
        std = df['regret'].std()
        print(f"{std:>15.6f}", end='')
    print()

    # Median regret
    print(f"{'Median regret':<35s}", end='')
    for df in methods.values():
        med = df['regret'].median()
        print(f"{med:>15.6f}", end='')
    print()

    # Max regret (worst decision)
    print(f"{'Max regret (worst decision)':<35s}", end='')
    for df in methods.values():
        max_reg = df['regret'].max()
        print(f"{max_reg:>15.6f}", end='')
    print()

    # Min regret (best decision)
    print(f"{'Min regret (best decision)':<35s}", end='')
    for df in methods.values():
        min_reg = df['regret'].min()
        print(f"{min_reg:>15.6f}", end='')
    print()

    # Average reward (not regret)
    print(f"{'Average reward obtained':<35s}", end='')
    for df in methods.values():
        avg_reward = df['chosen_reward'].mean()
        print(f"{avg_reward:>15.6f}", end='')
    print()

    print("=" * 90)

    # Improvement analysis
    if 'Random' in cum_regrets:
        print("\nIMPROVEMENT OVER RANDOM BASELINE:")
        print("-" * 90)
        random_cum = cum_regrets['Random']

        for name, cum in cum_regrets.items():
            if name != 'Random':
                improvement = ((random_cum - cum) / random_cum) * 100
                print(f"  {name:<20s}: {improvement:>6.2f}% reduction in cumulative regret")

    # Value of contextual features (LinUCB vs UCB1)
    if 'UCB1' in cum_regrets and 'LinUCB' in cum_regrets:
        print("\nVALUE OF FEATURES (LinUCB vs UCB1):")
        print("-" * 90)
        mab_cum = cum_regrets['UCB1']
        linucb_cum = cum_regrets['LinUCB']
        improvement = ((mab_cum - linucb_cum) / mab_cum) * 100
        print(f"  LinUCB improvement over UCB1: {improvement:>6.2f}%")

        if improvement > 0:
            print(f"  → Features are HELPFUL! LinUCB learns better with context. ✓")
        elif improvement < -5:
            print(f"  → Features are HARMFUL! LinUCB does worse with context. ✗")
        else:
            print(f"  → Features have MINIMAL impact. Context doesn't help much. ~")

    print("=" * 90)


def plot_cumulative_regret(results):
    """Plot cumulative regret for all methods"""
    methods = {k: v for k, v in results.items() if v is not None}

    if not methods:
        print("No results to plot!")
        return

    plt.figure(figsize=(14, 7))

    colors = {
        'LinUCB': '#2E86AB',
        'UCB1': '#A23B72',
        'Random': '#F18F01',
        'CF': '#4CAF50'
    }
    linestyles = {
        'LinUCB': '-',
        'UCB1': '-.',
        'Random': '--',
        'CF': ':'
    }

    for name, df in methods.items():
        plt.plot(
            df['cum_regret'],
            label=name,
            color=colors.get(name, 'gray'),
            linestyle=linestyles.get(name, '-'),
            linewidth=2.5 if name == 'LinUCB' else 2,
            alpha=0.9
        )

    plt.xlabel('Round (Recommendation)', fontsize=13)
    plt.ylabel('Cumulative Regret', fontsize=13)
    plt.title('Cumulative Regret Comparison: LinUCB vs UCB1 vs Random vs CF',
              fontsize=15, fontweight='bold', pad=20)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()

    output_path = OUT_DIR / "regret_comparison_all.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Cumulative regret plot saved to: {output_path}")

    plt.show()


def plot_regret_distribution(results):
    """Plot distribution of per-round regrets"""
    methods = {k: v for k, v in results.items() if v is not None}
    if not methods:
        print("No results to plot for regret distribution!")
        return

    n_methods = len(methods)
    # Up to 3 columns to keep it readable; more rows if needed
    n_cols = min(3, n_methods)
    n_rows = int(np.ceil(n_methods / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols + 1, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    colors = {
        'LinUCB': '#2E86AB',
        'UCB1': '#A23B72',
        'Random': '#F18F01',
        'CF': '#4CAF50'
    }

    for idx, (name, df) in enumerate(methods.items()):
        if idx >= len(axes):
            break
        ax = axes[idx]

        ax.hist(df['regret'], bins=50, color=colors.get(name, 'gray'),
                alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axvline(df['regret'].mean(), color='red', linestyle='--',
                   linewidth=2, label=f"Mean: {df['regret'].mean():.4f}")
        ax.set_xlabel('Regret', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{name} Regret Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3, axis='y')

    # Hide any unused axes (in case of 1 or 2 methods)
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    output_path = OUT_DIR / "regret_distribution_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Regret distribution plot saved to: {output_path}")

    plt.show()


def plot_learning_curves(results, window=1000):
    """Plot moving average of regret (learning curves)"""
    methods = {k: v for k, v in results.items() if v is not None}

    if not methods:
        print("No results to plot learning curves!")
        return

    plt.figure(figsize=(14, 7))

    colors = {
        'LinUCB': '#2E86AB',
        'UCB1': '#A23B72',
        'Random': '#F18F01',
        'CF': '#4CAF50'
    }

    for name, df in methods.items():
        ma_regret = df['regret'].rolling(window=window, min_periods=1).mean()

        plt.plot(ma_regret,
                 label=name,
                 color=colors.get(name, 'gray'),
                 linewidth=2,
                 alpha=0.9)

    plt.xlabel('Round', fontsize=13)
    plt.ylabel(f'Moving Average Regret (window={window})', fontsize=13)
    plt.title('Learning Curves: Is the Algorithm Improving Over Time?',
              fontsize=15, fontweight='bold', pad=20)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()

    output_path = OUT_DIR / "learning_curves_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Learning curves plot saved to: {output_path}")

    plt.show()


def main():
    print("\n" + "=" * 90)
    print("COMPREHENSIVE METHOD COMPARISON")
    print("=" * 90)

    # Load results
    print("\nLoading results...")
    results = load_results()

    # Print comparison table
    print_comparison_table(results)

    # Generate plots
    print("\nGenerating visualizations...")
    plot_cumulative_regret(results)
    plot_regret_distribution(results)
    plot_learning_curves(results)

    print("\n" + "=" * 90)
    print("COMPARISON COMPLETE")
    print("=" * 90)
    print(f"\nAll plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
