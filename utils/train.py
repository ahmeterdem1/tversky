from tqdm import tqdm
import torch
import numpy as np

def statistical_evaluation(
        model_class,
        optimizer_class,
        criterion,
        train_loader,
        test_loader,
        model_kwargs: dict = {},
        optimizer_kwargs: dict = {},
        rounds: int = 5,
        epochs: int = 5,
        device: str = "cpu"
):

    acc = []

    for r in range(rounds):
        print(f"Round {r + 1}/{rounds}")

        model = model_class(**model_kwargs).to(device)
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

        # Train the model
        train_losses = train(model, optimizer, criterion, train_loader, epochs, device)

        # Evaluate the model
        accuracy = eval(model, test_loader, device)
        acc.append(accuracy)

        print(f"Round {r + 1} - Loss: {np.mean(train_losses[-100:]):.4f}, Accuracy: {accuracy:.4f}")

    return np.mean(acc), np.std(acc)



def train(model, optimizer, criterion, train_loader, epochs: int = 5, device: str = "cpu"):

    losses = []
    model.train()

    for epoch in range(epochs):
        with tqdm(train_loader, unit="it") as loop:
            loop.set_description(f"Epoch {epoch + 1}")
            for i, (x, y) in enumerate(loop):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                if i % 10 == 0:
                    loop.set_postfix(loss=np.mean(losses[-10:]))

    return losses

def eval(model, test_loader, device: str = "cpu"):

    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for x, y in tqdm(test_loader):
            x, y = x.to(device), y.to(device)
            output = model(x)
            _, predicted = torch.max(output.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    return correct / total
