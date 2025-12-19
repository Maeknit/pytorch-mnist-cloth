from src.dataset import get_loaders

def test_get_loaders():
    train_loader, test_loader = get_loaders(batch_size=32)
    from torch.utils.data import DataLoader
    assert isinstance(train_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)

    # берём один batch и проверяем форму
    images, labels = next(iter(train_loader))
    # FashionMNIST: [batch_size, 1, 28, 28]
    assert images.ndim == 4
    assert images.shape[1] == 1
    assert images.shape[2] == 28
    assert images.shape[3] == 28

    # и что есть столько же меток, сколько картинок
    assert images.shape[0] == labels.shape[0]
