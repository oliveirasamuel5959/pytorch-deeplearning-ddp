import torch
import matplotlib.pyplot as plt

def prepare_image_to_plot(img, mean, std):
  # Reverse normalization
  mean_tensor = torch.tensor(mean).view(-1, 1, 1)
  std_tensor = torch.tensor(std).view(-1, 1, 1)

  img = img * std_tensor + mean_tensor

  # Clamp values to valid image range
  img = img.clamp(0, 1)

  # Convert (C, H, W) -> (H, W, C)
  img = img.permute(1, 2, 0)

  # If grayscale, remove channel dimension
  if img.shape[-1] == 1:
    img = img.squeeze(-1)


def emnist_visualization(images, labels, classes, mean, std):
  fig, axes = plt.subplots(4, 8, figsize=(10, 5))

  # Flatten 2D axes array
  axes = axes.flatten()

  for idx, ax in enumerate(axes):
    img_to_plot = prepare_image_to_plot(images[idx], mean, std)
    ax.imshow(img_to_plot, cmap='gray' if img_to_plot.ndim == 2 else None)

    # Convert numerical label to character
    label_idx = labels[idx].item()
    label = classes[label_idx]

    ax.set_title(f"Label: {label}\nIndex: {label_idx}")
    ax.axis("off")

  plt.tight_layout()
  plt.show()