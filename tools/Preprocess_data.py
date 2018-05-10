from os.path import join
from os import listdir

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np
from PIL import Image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir):
        super(DatasetFromFolder, self).__init__()
        self.image_path = join(image_dir)
        self.image_filenames = [x for x in listdir(self.image_path) if is_image_file(x)]

        self.reszie = transforms.Compose([transforms.RandomCrop((256,256))])

        dist_list = [transforms.CenterCrop((64,64)),
                     transforms.Scale((64//16, 64//16)),
                     transforms.Scale((64, 64),interpolation=Image.NEAREST)
                     ]
        self.dist_trans = transforms.Compose(dist_list)

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)



    def __getitem__(self, index):
        # Load Image
        img = load_img(join(self.image_path, self.image_filenames[index]))
        img = self.reszie(img)
        target = self.transform(img)

        disted_img = self.dist_trans(img)
        startx = 256 // 2 - (64 // 2)
        starty = 256 // 2 - (64 // 2)
        img.paste(disted_img, (startx, startx))

        #mask_img = Image.open("mask_cr.png").convert('L')
        #mask_img = mask_img.resize((256,256))
        #masked_img = Image.alpha_composite(img, mask_img)
        #img.paste(mask_img, (0, 0), mask_img)

        mask = torch.zeros(64 * 4, 64 * 4)
        mask_1 = torch.ones(64, 64)
        startx = 256 // 2 - (64 // 2)
        starty = 256 // 2 - (64 // 2)
        mask[startx:startx + 64, starty:starty + 64] = mask_1
        mask = torch.unsqueeze(mask, 0)

        input = self.transform(img)
        input_masked = torch.cat((input,mask),0)
        #save_img(input, "input.jpg")
        #save_img(target, "target.jpg")

        return input, target, input_masked

    def __len__(self):
        return len(self.image_filenames)


def get_training_set(root_dir):
    train_dir = join(root_dir, "train")
    return DatasetFromFolder(train_dir)


def get_test_set(root_dir):
    test_dir = join(root_dir, "test")
    return DatasetFromFolder(test_dir)