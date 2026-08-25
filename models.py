import torch
import torchvision.models as models

def load_model(model_name, num_classes, weights_file, device):
    """
    Instantiates the model architecture matching the saved checkpoint dimension
    before loading the weights cleanly.
    """
    model_name = model_name.lower().replace('-', '')
    
    if model_name == "resnet50":
        # Load an uninitialized ResNet50
        model = models.resnet50(weights=None)
        # Modify final linear layer to expect the 'saved_classes' dimension
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == "efficientnetb3":
        # Load an uninitialized EfficientNet-B3
        model = models.efficientnet_b3(weights=None)
        # Modify the classifier sequentially
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, num_classes)
        
    elif model_name == "visiontransformer":
        # Load standard Vit-b_16
        model = models.vit_b_16(weights=None)
        in_features = model.heads.head.in_features
        model.heads.head = torch.nn.Linear(in_features, num_classes)
        
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")

    # Load weights
    checkpoint = torch.load(weights_file, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model