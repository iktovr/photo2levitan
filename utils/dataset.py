from torch.utils.data import Dataset
import pathlib
import PIL
from joblib import Parallel, delayed
import os

from .utils import find_files, create_edges


class ImageDataset(Dataset):
    def __init__(self, root=None, transform=None, paths=None, extensions=['jpg', 'jpeg']):
        if paths is None:
            self.root = root
            self.paths = find_files(root, *extensions)
        else:
            self.paths = list(paths)
        self.transform = transform

    def load_image(self, index):
        image_path = self.paths[index]
        return PIL.Image.open(image_path).convert(mode='RGB')
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        img = self.load_image(index)

        if self.transform:
            return self.transform(img)
        else:
            return img


class ImageEdgeDataset(Dataset):
    def __init__(self, root, edges_root, transform=None):
        self.root = pathlib.Path(root)
        self.edges_root = pathlib.Path(edges_root)
        try:
            os.mkdir(edges_root)
        except FileExistsError:
            pass
        paths = list(self.root.glob("./**/*.jpg")) + list(pathlib.Path(root).glob("./**/*.jpeg"))
        self.paths = []
        for i, path in enumerate(paths):
            new_path = self.edges_root / f"{i}.jpg"
            self.paths.append((new_path, path))
            # create_edges(path, new_path)
        self.transform = transform
        Parallel(n_jobs=-1)(delayed(create_edges)(path, new_path) for (new_path, path) in self.paths)

    def load_image(self, index):
        path1, path2 = self.paths[index]
        return PIL.Image.open(path1).convert(mode='RGB'), PIL.Image.open(path2).convert(mode='RGB')
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        img1, img2 = self.load_image(index)

        if self.transform:
            return self.transform(img1), self.transform(img2)
        else:
            return img1, img2
