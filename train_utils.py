"""Shared training and testing utilities for the transformer assignment.

Provides train() and test() functions used across all notebook versions.
"""

import time

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def train(model, train_loader, optimiser, criterion, target_names, device, epochs=10):
    """Train the model and report per-epoch training and validation metrics.

    Iterates over train_loader for the number of epochs, updating weights via optimiser.
    A validation pass runs over the full training set after each epoch to compute validation loss and a classification report.

    Args:
        model (torch.nn.Module): Transformer encoder model to train.
        train_loader (DataLoader): PyTorch DataLoader for the training set.
        optimiser (torch.optim.Optimizer): Optimiser used to update model weights.
        criterion (torch.nn.Module): Loss function, applied to the positive class probability.
        target_names (list[str]): Class label strings for the classification report.
        device (str): Device to run training on ('cuda' or 'cpu').
        epochs (int): Number of full passes over the training set. Default is 10.

    Returns:
        dictionary: Contains the key 'training', with sub-keys:
            - 'epoch_1' ... 'epoch_N': classification_report dict per epoch.
            - 'training_loss': list of average training loss per epoch.
            - 'validation_loss': list of average validation loss per epoch.
    """
    loss_history = []
    val_loss_history = []
    result_dict = {}
    result_dict['training'] = {}
    counter = 0

    print(f"Performing the training phase..")

    start_time = time.time()
    # Training phase -------------------
    for epoch in range(1, epochs+1):
        model.train() # sets the neural network module and all its sub-modules into training mode
        epoch_loss = 0.0 # initialise the epoch_loss

        for batch in train_loader:
            counter += 1
            # extract the data (per row)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimiser.zero_grad() # zeroing the gradient

            probabilities = model(input_ids, attention_mask) # call the model directly
            positive_probs = probabilities[:, 1] # positive class
            loss = criterion(positive_probs, labels.float()) # averaging the loss automatically

            loss.backward() # computes the derivative (gradient) of the loss with respect to all model parameters
            optimiser.step() # updates the model's parameters (weights and biases) using the gradients computed during the backward pass

            epoch_loss += loss.item() * input_ids.size(0) # convert the average back to sum

        avg_loss = epoch_loss / len(train_loader.dataset) # calculate the avg per datapoint
        loss_history.append(avg_loss)


        # Validation phase -------------------
        model.eval() # sets the neural network module and all its sub-modules into evaluation/testing mode
        val_loss = 0.0
        y_true_list, y_pred_list = [], []

        with torch.no_grad():
            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                probabilities = model(input_ids, attention_mask)
                positive_probs = probabilities[:, 1] # recompute
                predictions = probabilities.argmax(dim=1)

                loss = criterion(positive_probs, labels.float())
                val_loss += loss.item() * input_ids.size(0)

                y_true_list.extend(labels.cpu().tolist()) # move the variable to CPU for report and visualisation
                y_pred_list.extend(predictions.cpu().tolist()) # move the variable to CPU for report and visualisation


        avg_val_loss = val_loss / len(train_loader.dataset)
        val_loss_history.append(avg_val_loss)

        # store the result to dict
        training_report = classification_report(y_true_list, y_pred_list, target_names=target_names)
        result_dict['training'][f"epoch_{epoch}"] = classification_report(y_true_list, y_pred_list, target_names=target_names, output_dict=True) # store the classification report per epoch

        print(f"Epoch: {epoch}/{epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    training_report = classification_report(y_true_list, y_pred_list, target_names=target_names) # show the report 1 time only, at the end
    conf_matrix = confusion_matrix(y_true_list, y_pred_list)
    print(f"The classification report is:")
    print(f"-" * 60)
    print(training_report) # display the training report

    print(f"The confusion matrix is:")
    print(f"-" * 60)
    print(conf_matrix) # display the configuration matrix

    print(f"Displaying the Learning Curve and Confusion Matrix")
    plt.figure(figsize=(15, 5))
    plt.title("Learning Curve")
    plt.plot(loss_history, color='navy', linewidth=2, marker='o', label="Training Loss")
    plt.plot(val_loss_history, color='red', linewidth=2, marker='x', label="Validation Loss")
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()
    plt.show() # shot the learning curve plot

    end_time = time.time()
    elapsed_time = end_time - start_time # calculate the elapsed time
    display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=target_names)
    display.plot(cmap=plt.cm.Blues)
    plt.show() # show the confusion matrix

    print(f"\nThe training is completed!")
    print(f"\nElapsed training time is: {elapsed_time} seconds ({elapsed_time/60:.2f} minutes)")

    # store the training loss and validation loss
    result_dict['training']['training_loss'] = loss_history
    result_dict['training']['validation_loss'] = val_loss_history
    return result_dict


def test(model, test_loader, target_names, device, result_dict):
    """Evaluate the model on the test set and report classification metrics.

    Runs a single forward pass over all samples in test_loader. No gradient computation or weight updates.

    Args:
        model (torch.nn.Module): Trained transformer encoder model.
        test_loader (DataLoader): PyTorch DataLoader for the test set.
        target_names (list[str]): Class label strings for the classification report.
        device (str): Device to run evaluation on ('cuda' or 'cpu').
        result_dict (dictionary): Existing result dictionary to merge the test report into.

    Returns:
        dictionary: The input result_dict, with a 'test' key added containing the classification_report output_dict (per-class precision, recall, f1-score, support, 'accuracy', 'macro avg', and 'weighted avg').
    """
    model.eval()
    y_pred_list, y_true_list = [], []


    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            probabilities = model(input_ids, attention_mask)
            predictions = probabilities.argmax(dim=1)

            y_true_list.extend(labels.cpu().tolist()) # move the variable to CPU for report and visualisation
            y_pred_list.extend(predictions.cpu().tolist()) # move the variable to CPU for report and visualisation

    test_report = classification_report(y_true_list, y_pred_list, target_names=target_names)
    conf_matrix = confusion_matrix(y_true_list, y_pred_list)

    print(f"The classification report is:")
    print(f"-"*60)
    print(test_report)

    print(f"\nThe confusion matrix is:")
    print(f"-"*60)
    print(conf_matrix)

    display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=target_names)
    display.plot(cmap=plt.cm.Blues)
    plt.show()

    print(f"\nThe testing is completed!")

    # store the result to dict
    result_dict['test'] = classification_report(y_true_list, y_pred_list, target_names=target_names, output_dict=True)
    return result_dict
