"""
Plot mAP50-95 per epoch from YOLO training results.csv.

Reads results.csv from YOLO training and creates a graph showing
mAP50-95 metric progression across epochs.
"""
import csv
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np


def load_results_csv(csv_path):
    """Load results.csv and extract epoch and mAP50-95 data."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")
    
    epochs = []
    map50_95 = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epoch = int(row['epoch'])
                # Column name is 'metrics/mAP50-95(B)' in YOLO results
                map_val = float(row.get('metrics/mAP50-95(B)', row.get('metrics/mAP50-95', '')))
                epochs.append(epoch)
                map50_95.append(map_val)
            except (ValueError, KeyError) as e:
                print(f"Warning: Skipping row {len(epochs)+1}: {e}")
                continue
    
    if not epochs:
        raise ValueError("No valid data found in CSV file")
    
    return epochs, map50_95


def plot_map50_95(epochs, map50_95, output_path=None, title=None):
    """Prettier mAP50-95 plot."""
    plt.figure(figsize=(12, 6))

    # cleaner line
    plt.plot(
        epochs,
        map50_95,
        linewidth=3,
        color="#7C878E",   # nicer blue
        label="mAP50-95"
    )

    # highlight best point
    best_idx = np.argmax(map50_95)
    best_epoch = epochs[best_idx]
    best_map = map50_95[best_idx]

    plt.scatter(
        [best_epoch],
        [best_map],
        color="#F76902",
        s=80,
        zorder=5
    )

    plt.text(
        best_epoch,
        best_map + 0.01,
        f"best: {best_map:.3f} @ epoch {best_epoch}",
        color="#F76902",
        fontsize=11,
        ha="center"
    )

    # labels
    plt.xlabel("epoch", fontsize=14, fontweight="bold")
    plt.ylabel("mAP50-95", fontsize=14, fontweight="bold")

    # title
    if title:
        plt.title(title, fontsize=16, fontweight="bold")
    else:
        plt.title("YOLO mAP50-95 over epochs", fontsize=16, fontweight="bold")

    # grid
    plt.grid(True, alpha=0.25, linestyle="--")

    # remove ugly top/right borders
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    # legend
    plt.legend(frameon=False, fontsize=12)

    plt.tight_layout()

    # save
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()

    plt.close()

    # summary
    print("\nSummary:")
    print(f"  total epochs: {len(epochs)}")
    print(f"  best mAP50-95: {best_map:.4f} at epoch {best_epoch}")
    print(f"  final mAP50-95: {map50_95[-1]:.4f} at epoch {epochs[-1]}")
    print(f"  improvement: {map50_95[-1] - map50_95[0]:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot mAP50-95 per epoch from YOLO training results.csv"
    )
    parser.add_argument("--csv", default="results.csv",
                       help="Path to results.csv file (default: results.csv in current directory)")
    parser.add_argument("--output", "-o", default=None,
                       help="Output plot path (default: shows plot interactively)")
    parser.add_argument("--title", default=None,
                       help="Plot title (default: auto-generated)")
    
    args = parser.parse_args()
    
    try:
        # Load data
        print(f"Loading results from: {args.csv}")
        epochs, map50_95 = load_results_csv(args.csv)
        print(f"Loaded {len(epochs)} epochs")
        
        # Generate output path if not specified
        if args.output is None:
            csv_dir = os.path.dirname(args.csv) if os.path.dirname(args.csv) else '.'
            args.output = os.path.join(csv_dir, "map50_95_plot.png")
        
        # Plot
        plot_map50_95(epochs, map50_95, args.output, args.title)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

