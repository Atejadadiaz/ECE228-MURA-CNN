import torch.nn as nn
import torchvision.models as models

def get_model(model_name='resnet18', weights='IMAGENET1K_V1'):
    """
    Returns a CNN model adapted for binary classification.

    Parameters:
    - model_name: str, one of ['resnet18', 'resnet50', 'densenet121', 'convnext_tiny']
    - weights: str or None, torchvision pretrained weights ID (e.g., 'IMAGENET1K_V1')

    Returns:
    - model: nn.Module
    """
    if model_name == 'resnet18':
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 1)

    elif model_name == 'resnet50':
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 1)

    elif model_name == 'densenet121':
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, 1)

    elif model_name == 'convnext_tiny':
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 1)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model
