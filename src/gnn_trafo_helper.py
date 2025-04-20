import awkward
import torch
import numpy as np

def compute_norm_stats(dataset):
    """
    Compute the global mean & std for each feature (t, x, y)
    and for the labels (xpos, ypos) on the training set.
    Returns a dict of six scalars.
    """
    # flatten feature ragged arrays
    t_flat = awkward.flatten(dataset["data"][:, 0, :])
    x_flat = awkward.flatten(dataset["data"][:, 1, :])
    y_flat = awkward.flatten(dataset["data"][:, 2, :])

    stats = {
        "t_mean": awkward.mean(t_flat),
        "t_std":  awkward.std(t_flat),
        "x_mean": awkward.mean(x_flat),
        "x_std":  awkward.std(x_flat),
        "y_mean": awkward.mean(y_flat),
        "y_std":  awkward.std(y_flat),
    }
    # labels
    stats["xpos_mean"] = awkward.mean(dataset["xpos"])
    stats["xpos_std"]  = awkward.std(dataset["xpos"])
    stats["ypos_mean"] = awkward.mean(dataset["ypos"])
    stats["ypos_std"]  = awkward.std(dataset["ypos"])
    return stats

def normalize_dataset(dataset, stats):
    """
    Given any split (train/val/test) and the stats dict from compute_norm_stats,
    normalize data["data"] in-place and return the dataset.
    """
    # extract the three features
    times = dataset["data"][:, 0:1, :]
    xs = dataset["data"][:, 1:2, :]
    ys = dataset["data"][:, 2:3, :]

    # normalize each
    norm_times = (times - stats["t_mean"]) / stats["t_std"]
    norm_x = (xs - stats["x_mean"]) / stats["x_std"]
    norm_y = (ys - stats["y_mean"]) / stats["y_std"]

    # stack them back into (3, n_hits)
    dataset["data"] = awkward.concatenate([norm_times, norm_x, norm_y], axis=1)

    # normalize labels
    dataset["xpos"] = (dataset["xpos"] - stats["xpos_mean"]) / stats["xpos_std"]
    dataset["ypos"] = (dataset["ypos"] - stats["ypos_mean"]) / stats["ypos_std"]

    return dataset


def denorm_preds_and_labels(preds_norm, labels_norm, stats):
    """
    Denormalize both predictions and labels
    based on the stats dict from compute_norm_stats.

    preds_norm and labels_norm are assumed to be torch tensors.

    Returns numpy arrays of shape (n_samples, n_labels).
    """
    preds_denorm = np.zeros_like(preds_norm.numpy())
    labels_denorm = np.zeros_like(labels_norm.numpy())

    # xpos
    preds_denorm[:,0] = preds_norm[:,0] * stats["xpos_std"] + stats["xpos_mean"]
    labels_denorm[:,0] = labels_norm[:,0] * stats["xpos_std"] + stats["xpos_mean"]
    # ypos
    preds_denorm[:,1] = preds_norm[:,1] * stats["ypos_std"] + stats["ypos_mean"]
    labels_denorm[:,1] = labels_norm[:,1] * stats["ypos_std"] + stats["ypos_mean"]

    return preds_denorm, labels_denorm


def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=50):
    """
    Train a given model using the provided training and validation data loaders.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to be trained.
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training dataset.
    val_loader : torch.utils.data.DataLoader
        DataLoader for the validation dataset.
    optimizer : torch.optim.Optimizer
        Optimizer used for updating model parameters.
    criterion : callable
        Loss function used to compute the training and validation loss.
    epochs : int, optional
        Number of epochs to train the model (default is 50).

    Returns
    -------
    tuple of lists
        A tuple containing two lists:
        - train_loss: List of training losses for each epoch.
        - val_loss: List of validation losses for each epoch.
    """
    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        for data, labels in train_loader:
            optimizer.zero_grad()
            preds = model(data)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)
        train_loss = running_loss / total
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_running = 0.0
        total_val = 0
        with torch.no_grad():
            for data, labels in val_loader:
                preds = model(data)
                val_running += criterion(preds, labels).item() * labels.size(0)
                total_val += labels.size(0)
        val_loss = val_running / total_val
        val_losses.append(val_loss)

        print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return train_losses, val_losses

def evaluate_model(model, test_loader, criterion):
    """
    Evaluate the model on the test dataset.

    Parameters
    ----------
    model : torch.nn.Module
        The trained neural network model to be evaluated.
    test_loader : torch.utils.data.DataLoader
        DataLoader for the test dataset.
    criterion : callable
        Loss function used to compute the test loss.

    Returns
    -------
    tuple
        A tuple containing:
        - all_preds (torch.Tensor): Concatenated predictions for all test samples.
        - all_true (torch.Tensor): Concatenated true labels for all test samples.
        - test_loss (float): Average test loss over the entire test dataset.
    """
    print("Evaluating model on test dataset...")
    model.eval()
    all_preds = []
    all_true = []
    test_loss = 0.0
    total = 0

    with torch.no_grad():
        for data, labels in test_loader:
            preds = model(data)
            loss = criterion(preds, labels)
            test_loss += loss.item() * labels.size(0)
            total += labels.size(0)

            all_preds.append(preds)
            all_true.append(labels)

    # Concatenate all predictions and true labels
    all_preds = torch.cat(all_preds, dim=0)
    all_true = torch.cat(all_true, dim=0)

    test_loss /= total
    print(f"Test Loss: {test_loss:.4f}")
    return all_preds, all_true, test_loss