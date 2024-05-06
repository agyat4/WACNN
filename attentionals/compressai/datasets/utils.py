# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
class ImageFolder(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split #/ "data"

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]
        print("the size of the sample",len(self.samples))

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """

        #img = Image.open(self.samples[index])#.convert("RGB")

        img= np.load(self.samples[index])




        #img=np.transpose(img,(1,2,0))

        #print(img.dtype)
        #print(img.shape)

        if self.transform:
            images = []
            #i=0
            #pltimg=img
            for image in img:

                image[image < 1] = 1
                image = np.log2(image)
                #image *= (255.0 / image.max())
                image = ((image - image.min()) * (255.0 / (image.max() - image.min())))
                image = image / 255.0
                image=image.astype(np.float32)
                #PIL_image=Image.fromarray(image.astype('uint8'))



                if image.ndim == 3 and image.shape[0] == 1:
                    image = image.squeeze(0)

                image = np.array(image)

                images.append(image)
                #i+=1
            #print("shape of images",len(images))
            #print("the value of i ", i)
            img = np.stack(images, axis=0)

            img = np.transpose(img, (1, 2, 0))
            transformed = self.transform(img)
            #tramsformed = np.transpose(transformed,(2,0,1))

            #print("shape of images after transform", transformed.shape)

            # Number of columns equals the number of images

            # Create a figure and axes
            '''fig, axes = plt.subplots(1,9,figsize=(40, 16))

            # Loop through each image and plot it
            for i in range(9):

                axes[i].imshow(transformed[i], cmap="gray")
                axes[i].set_title(i)
                

                axes[i].axis("off")

            plt.tight_layout()
            plt.show()
            exit()'''
            return transformed

        return img

    def __len__(self):
        return len(self.samples)
