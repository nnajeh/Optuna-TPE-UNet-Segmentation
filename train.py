from libraries import *
from model import *
from objective import *




# Instantiate the model with optimized `n_filters`
model = UNet(in_channels=1, out_channels=1, n_filters=73).to(device)

# Optimizer with optimized learning rate
optimizer = optim.Adam(model.parameters(), lr=0.0017754230774176293)

# Loss functions
loss_fn_1 = nn.BCEWithLogitsLoss()  # Add your custom loss functions if needed
loss_fn_2 = FocalLoss()
loss_fn_3 = DiceLoss()

# Set number of epochs
epochs = 20000

# Training function
def train(model, train_loader, optimizer, loss_fn_1, loss_fn_2, loss_fn_3, device):
    model.train()
    total_train_loss = 0

    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss_1 = loss_fn_1(y_pred, y)
        loss_2 = loss_fn_2(y_pred, y)
        loss_3 = loss_fn_3(y_pred, y)
        loss = loss_1 + loss_2 + loss_3

        # 3. Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()  # Accumulate the loss

    return total_train_loss / len(train_loader)  # Return average loss per batch

# Evaluation function for Dice score
def evaluate(model, val_loader, device):
    model.eval()
    dice_score = 0

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            y_pred = model(X)

            # 2. Apply sigmoid to get predictions in the range [0, 1]
            preds = torch.sigmoid(y_pred)

            # 3. Calculate Dice score
            preds = (preds > 0.5).float()  # Binarize predictions
            intersection = (preds * y).sum(dim=(2, 3))
            union = preds.sum(dim=(2, 3)) + y.sum(dim=(2, 3))
            dice_batch = 2 * intersection / (union + 1e-7)
            dice_score += dice_batch.mean().item()

    return dice_score / len(val_loader)  # Return average Dice score

# Visualization function
def visualize_predictions(model, val_loader, device):
    model.eval()
    batch = next(iter(val_loader))
    X, y = batch
    X, y = X.to(device), y.to(device)

    with torch.no_grad():
        # Forward pass
        y_pred = model(X)
        preds = torch.sigmoid(y_pred)
        preds = (preds > 0.5).float()

    # Visualize the first 3 images, ground truth, and predictions
    for i in range(3):
        plt.figure(figsize=(15, 5))

        # Input Image
        plt.subplot(1, 3, 1)
        plt.imshow(X[i].cpu().squeeze(), cmap='gray')
        plt.title("Input Image")
        plt.axis('off')

        # Ground Truth Mask
        plt.subplot(1, 3, 2)
        plt.imshow(y[i].cpu().squeeze(), cmap='gray')
        plt.title("Ground Truth Mask")
        plt.axis('off')

        # Predicted Mask
        plt.subplot(1, 3, 3)
        plt.imshow(preds[i].cpu().squeeze(), cmap='gray')
        plt.title("Predicted Mask")
        plt.axis('off')

        plt.show()

# Track training and validation losses
train_total_losses = []
val_total_losses = []

# Create training and testing loop
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch + 1}/{epochs}")

    ### Training
    train_loss = train(model, train_loader, optimizer, loss_fn_1, loss_fn_2, loss_fn_3, device)

    ### Evaluation
    test_loss = evaluate(model, val_loader, device)

    train_total_losses.append(train_loss)
    val_total_losses.append(test_loss)

    if epoch % 5 == 0:
        print(f"Train loss: {train_loss:.5f} | Test loss: {test_loss:.5f}")

        # Visualization
        visualize_predictions(model, val_loader, device)

    # Save model and plot losses every 500 epochs
    if epoch % 500 == 0 and epoch != 0:
        torch.save(model.state_dict(), f"./model-{epoch}.pth")
        plt.figure(figsize=(20, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_total_losses, label='Train Loss')
        plt.plot(val_total_losses, label='Val Loss')
        plt.title("Training & Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()



# Optimize using Optuna with TPE
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Optuna study
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=50, timeout=3600)

    # Display the best hyperparameters
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
