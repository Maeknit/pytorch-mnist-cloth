from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2859,), (0.3530,))  # mnist-cloth
])

def get_loaders(batch_size=64):
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Показать 10 примеров
def visualize_samples(loader, num_samples=10):
    data_iter = iter(loader)
    images, labels = next(data_iter)

    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        img = images[i].squeeze()
        label = labels[i].item()
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(classes[label])
        axes[i].axis('off')
    plt.show()

if __name__ == '__main__':
    train_loader, _ = get_loaders()
    visualize_samples(train_loader)