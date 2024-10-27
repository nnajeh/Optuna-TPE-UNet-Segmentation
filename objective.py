# Import libraries
from libraries import *
from model import *


# Define the objective function for Optuna
def objective(trial):
    # Hyperparameters to tune
    n_filters = trial.suggest_int("n_filters", 32, 128)
    learning_rate = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32,64])
    
    # Model
    model = UNet().to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Loss function
    criterion = dice_loss
    
    # Data preparation 
    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    # Create DataLoaders for both training and validation datasets
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_dice_score = evaluate(model, val_loader, device)
        
        # Use the negative Dice score as the optimization target
        trial.report(-val_dice_score, epoch)
        
        # If the intermediate result is not good, prune the trial
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return -val_dice_score
