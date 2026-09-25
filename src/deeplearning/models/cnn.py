import torch
import torch.nn as nn


class ConvBlock(nn.Module):
  def __init__(self, input_channels, out_channels):
    super(ConvBlock, self).__init__()
    
    self.block = nn.Sequential(
      nn.Conv2d(input_channels, out_channels, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
    )
    
  def forward(self, x):
    return self.block(x)
  
  
class CNNClassifier(nn.Module):
  def __init__(self, input_channels, num_classes):
    super(CNNClassifier, self).__init__()
    
    self.features = nn.Sequential(
      ConvBlock(input_channels, out_channels=32),
      ConvBlock(input_channels=32, out_channels=64),
      ConvBlock(input_channels=64, out_channels=128),
    )
    
    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Linear(128 * 4 * 4, 256),  # Assuming input image size is 32x32
      nn.ReLU(),
      nn.Linear(256, num_classes),
    )
    
  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x)
    return x
  
def main():
  model = CNNClassifier(input_channels=3, num_classes=10)
  
  print("\n ====== [MODEL] Architecture =======")
  print(model)
  
  print("\n ====== [MODEL] Layers details =======")
  for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
  
  total_params = sum(param.numel() for param in model.parameters())
  total_params_M = total_params / 1e6
  print(f"\n[MODEL] Total parameters: {total_params_M:.2f}M")
  
  print(f"\n[MODEL] Size in memory assuming 32-bit float precision")
  model_size_mb = (total_params * 4) / (1024 ** 2)
  print(f"Model size: {model_size_mb:.2f} MB")
  
  print("\n[MODEL] ======= SUMMARY ========")
  # Iterate through the main blocks
  for name, block in model.named_children():
    print(f"Block {name} has a total of {len(list(block.children()))} layers:")

    # List all children layers in the block
    for idx, layer in enumerate(block.children()):
      # Check if the layer is terminal (no children) or not
      if len(list(layer.children())) == 0:
        print(f"\t {idx} - Layer {layer}")
        
      # If the layer has children, it's a sub-block, then print only the number of children and its name
      else:
        layer_name = layer._get_name()  # More user-friendly name
        print(f"\t {idx} - Sub-block {layer_name} with {len(list(layer.children()))} layers")    
        
  print(f"\n[MODEL] ======= Zoom into ConvBlock model ========")
  first_conv_module = model.features[0]

  for idx, module in enumerate(first_conv_module.modules()):
    # Avoid printing the top-level module itself
    if idx > 0 :
      print(module)
  
  
  print(f"\n[MODEL] ======= Layers type Conv2d counts ========")
  type_layer = nn.Conv2d
  selected_layers = [layer for layer in model.modules() if isinstance(layer, type_layer)]
  print(f"Number of {type_layer.__name__} layers: {len(selected_layers)}")
  
if __name__ == '__main__':
  main()