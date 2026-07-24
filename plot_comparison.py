import os
import pandas as pd
import matplotlib.pyplot as plt

# Create output directory
os.makedirs("outputs/comparison", exist_ok=True)

# ----------------------------------------
# Model specifications
# ----------------------------------------
model_info = [
    {"dir": "resnet50", "name": "ResNet-50", "params": 25.6},
    {"dir": "densenet121", "name": "DenseNet-121", "params": 8.0},
    {"dir": "efficientnet_b0", "name": "EfficientNet-B0", "params": 5.3},
    {"dir": "vgg16", "name": "VGG-16", "params": 138.3},
    {"dir": "vit", "name": "ViT", "params": 86.6},
    {"dir": "swin", "name": "Swin", "params": 28.3}
]

names = []
params = []
val_losses = []
test_accs = []

for m in model_info:
    hist_path = os.path.join("outputs", m["dir"], "history.csv")
    metric_path = os.path.join("outputs", m["dir"], "metrics.csv")
    
    val_loss = 0.0
    test_acc = 0.0
    
    if os.path.exists(hist_path):
        df_h = pd.read_csv(hist_path)
        if "val_loss" in df_h.columns:
            val_loss = df_h["val_loss"].min()
            
    if os.path.exists(metric_path):
        df_m = pd.read_csv(metric_path)
        if "Accuracy" in df_m.columns:
            test_acc = df_m["Accuracy"].iloc[0] * 100.0

    names.append(m["name"])
    params.append(m["params"])
    val_losses.append(val_loss)
    test_accs.append(test_acc)

# Plot 1: Validation Loss vs Model Parameters
plt.figure(figsize=(9, 6))
plt.scatter(
    params,
    val_losses,
    s=180,
    edgecolors="black",
    linewidth=1.2,
    c="skyblue",
    zorder=3
)

for x, y, label in zip(params, val_losses, names):
    plt.annotate(
        label,
        (x, y),
        xytext=(8, 6),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold"
    )

plt.xscale("log")
plt.xlabel("Number of Parameters (Millions)", fontsize=13)
plt.ylabel("Minimum Validation Loss", fontsize=13)
plt.title("Validation Loss vs Model Parameters", fontsize=16)
min_y = min(val_losses) - 0.05 if val_losses else 0
max_y = max(val_losses) + 0.05 if val_losses else 1.0
plt.ylim(max(0, min_y), max_y)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("outputs/comparison/loss_vs_parameters.png", dpi=300, bbox_inches="tight")
plt.close()

# Plot 2: Test Accuracy Comparison Bar Chart
plt.figure(figsize=(10, 6))
bars = plt.bar(names, test_accs, color=["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"])
plt.ylabel("Test Accuracy (%)", fontsize=13)
plt.title("Test Accuracy Comparison Across Models", fontsize=16)
plt.ylim(0, 105)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("outputs/comparison/accuracy_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

print("Comparison plots saved to outputs/comparison/")