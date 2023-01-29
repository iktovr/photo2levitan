import random
import torch


class ImagePool:
    def __init__(self, size):
        self.size = size
        self.images = []

    def query(self, images):
        if self.size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if len(self.images) < self.size:
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.random()
                if p < 0.5:
                    ind = random.randrange(self.size)
                    return_images.append(self.images[ind].clone())
                    self.images[ind] = image
                else:
                    return_images.append(image)
        return torch.cat(return_images, 0)
