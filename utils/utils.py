import torch.nn as nn
import torch
import pathlib
import PIL
import cv2 as cv


def init_weights(net, std):
    def init_func(module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            torch.nn.init.normal_(module.weight, 0, std)
    net.apply(init_func)


def find_files(root, *extensions):
    files = []
    for ext in extensions:
        files += pathlib.Path(root).glob(f"./**/*.{ext}")
    return files


def create_edges(file, edges_file):
    img = cv.imread(str(file))
    edges = PIL.Image.fromarray(cv.Canny(img, 100, 200)).convert('RGB')
    edges = PIL.ImageOps.invert(edges)
    edges.save(str(edges_file))
