from src.dataset import get_loaders
import torch
from torch.utils.data import DataLoader

def test_get_loaders_basic():
    train_loader, test_loader = get_loaders(batch_size=32)

    # Инициализация
    assert isinstance(train_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)

    # Есть данные?
    assert len(train_loader) > 0
    assert len(test_loader) > 0

def test_train_batch_shape_and_types():
    batch_size = 32
    train_loader, _ = get_loaders(batch_size=batch_size)

    images, labels = next(iter(train_loader))

    assert images.ndim == 4
    assert images.shape[0] == batch_size
    assert images.shape[1] == 1
    assert images.shape[2] == 28
    assert images.shape[3] == 28

    assert isinstance(images, torch.Tensor)
    assert isinstance(labels, torch.Tensor)

    assert images.dtype == torch.float32
    assert labels.dtype in (torch.int64, torch.long)

    assert images.shape[0] == labels.shape[0]

def test_multiple_batches_iteration():
    batch_size = 16
    train_loader, _ = get_loaders(batch_size=batch_size)

    # Пройдемся по нескольким батчам, чтобы убедиться, что итерация стабильная
    num_batches_to_check = 3
    for i, (images, labels) in enumerate(train_loader):
        # Проверяем только первые несколько батчей, чтобы не тормозить CI
        if i >= num_batches_to_check:
            break

        assert images.ndim == 4
        assert images.shape[1:] == (1, 28, 28)
        assert images.shape[0] == labels.shape[0]
