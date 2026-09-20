"""
What IS an image, to a computer?
A quick look at the raw data before any training happens.
"""
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

image, label = train_data[0]

print("What type of object is 'image'?", type(image))
print("Its shape:", image.shape)          # [channels, height, width]
print("Smallest pixel value:", image.min().item())
print("Largest pixel value:", image.max().item())
print("The label (correct answer):", label)

print("\nMiddle 5x5 of the image (where the digit actually is):")
print(image[0, 12:17, 12:17])

plt.imshow(image[0], cmap="gray")
plt.title(f"This is a {label}")
plt.show()
