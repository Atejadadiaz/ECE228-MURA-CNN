import torch
import torch.nn as nn
from torchmetrics import CohenKappa
from src.model import get_model
from src.data import get_dataloaders  # Asegúrate de tener esta función en data.py


def trainer(model, train_loader, val_loader, learning_rate, step_size, epochs, output_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    kappa_metric = CohenKappa(num_classes=2, task='binary', weights='quadratic').to(device)

    for epoch in range(1, epochs + 1):
        # -------- TRAINING --------
        model.train()
        train_loss, train_total, train_correct = 0.0, 0, 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1).float()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = (torch.sigmoid(outputs) >= 0.5).int()
            train_correct += (preds == labels.int()).sum().item()
            train_total += labels.size(0)
            train_loss += loss.item() * labels.size(0)

        scheduler.step()
        train_acc = train_correct / train_total
        train_loss /= train_total

        # -------- VALIDATION --------
        model.eval()
        val_loss, val_total, val_correct = 0.0, 0, 0
        kappa_metric.reset()

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).unsqueeze(1).float()

                outputs = model(images)
                loss = criterion(outputs, labels)
                preds = (torch.sigmoid(outputs) >= 0.5).int()

                val_correct += (preds == labels.int()).sum().item()
                val_total += labels.size(0)
                val_loss += loss.item() * labels.size(0)

                kappa_metric.update(preds.cpu(), labels.cpu().int())

        val_acc = val_correct / val_total
        val_loss /= val_total
        val_kappa = kappa_metric.compute().item()

        print(f"[Epoch {epoch:02d}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Kappa: {val_kappa:.4f}")

    return model

# ========================
# Main script to run training
# ========================
if __name__ == "__main__":
    # Path
    data_dir = "data/MURA-v1.1"

    #Dataloaders
    train_loader, val_loader = get_dataloaders(data_dir, data_dir)

    # Load model (choose one: 'resnet18', 'resnet50', 'densenet121', 'convnext_tiny')
    model = get_model('resnet18')

    # Train
    trained_model = trainer(
        model,
        train_loader,
        val_loader,
        learning_rate=1e-3,
        step_size=5,
        epochs=10,
        output_rate=1
    )

    # Save weights
    torch.save(trained_model.state_dict(), "results/resnet18_mura.pth")
