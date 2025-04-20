import os
import numpy as np
import awkward
from gnn_encoder import GNNEncoder, collate_fn_gnn
from gnn_trafo_helper import train_model, evaluate_model, compute_norm_stats, normalize_dataset, denorm_preds_and_labels
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

DATA_PATH = "data"

# Load the dataset
train_dataset = awkward.from_parquet(os.path.join(DATA_PATH, "train.pq"))
val_dataset = awkward.from_parquet(os.path.join(DATA_PATH, "val.pq"))
test_dataset = awkward.from_parquet(os.path.join(DATA_PATH, "test.pq"))

# Normalize data and labels
stats = compute_norm_stats(train_dataset)
train_dataset = normalize_dataset(train_dataset, stats)
val_dataset = normalize_dataset(val_dataset, stats)
test_dataset = normalize_dataset(test_dataset, stats)

# batch size and create DataLoader objects 
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_gnn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_gnn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_gnn)

# Define the model, optimizer, and loss function
model = GNNEncoder(in_channels=3, hidden_channels=64, out_channels=2, k=12, n_layers=4)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Train the model
train_loss, val_loss = train_model(model, train_loader, val_loader, optimizer, criterion, epochs=50)

# Evaluate the model on the test set
preds, true, test_loss = evaluate_model(model, test_loader, criterion)

# save the model
torch.save(model.state_dict(), "gnn_model.pth") 

# denormalize the predictions and labels
preds_denorm, true_denorm = denorm_preds_and_labels(preds, true, stats)

# plot the training and validation loss
plt.figure()
plt.plot(train_loss, label="Training loss")
plt.plot(val_loss, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.title("Training and Validation Loss")
plt.savefig("figures/train_val_loss.png")


# plot the predictions vs true labels
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# ─── xpos ───
gt_x = true_denorm[:, 0]
pd_x = preds_denorm[:, 0]
axes[0].scatter(gt_x, pd_x, s=6, alpha=0.4)
lo_x, hi_x = gt_x.min(), gt_x.max()
axes[0].plot([lo_x, hi_x], [lo_x, hi_x],
             c='black', linestyle='--', label='Perfect prediction')
axes[0].set_xlabel("True xpos")
axes[0].set_ylabel("Predicted xpos")
axes[0].legend()
axes[0].grid(color='grey', linestyle=':', linewidth=0.5)
# ─── ypos ───
gt_y = true_denorm[:, 1]
pd_y = preds_denorm[:, 1]
axes[1].scatter(gt_y, pd_y, s=6, alpha=0.4, color='orange')
lo_y, hi_y = gt_y.min(), gt_y.max()
axes[1].plot([lo_y, hi_y], [lo_y, hi_y],
             c='black', linestyle='--', label='Perfect prediction')
axes[1].set_xlabel("True ypos")
axes[1].set_ylabel("Predicted ypos")
axes[1].legend()
axes[1].grid(color='grey', linestyle=':', linewidth=0.5)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("figures/predictions_vs_true_improv.png")


# plot error histogram
errors = np.linalg.norm(true_denorm - preds_denorm, axis=1)
mean_err = errors.mean()
std_err  = errors.std()
fig, ax = plt.subplots()
ax.hist(errors, bins=50, alpha=0.7)
ax.set_xlabel("Distance error [m]")
ax.set_ylabel("Counts")
ax.set_title("Error Histogram")
# textbox
textstr = f"Mean = {mean_err:.2f}\nStd  = {std_err:.2f}"
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
ax.text(0.95, 0.95, textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props)
plt.savefig("figures/error_histogram.png")
plt.show()