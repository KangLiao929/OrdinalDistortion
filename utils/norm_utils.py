def get_lr(epoch, lr):
    return lr * (0.98 ** epoch)