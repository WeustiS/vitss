import torch
from vit_pytorch import ViT
import glob
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

xglob = glob.glob('320/*.jpg')
yglob = glob.glob('640/*.jpg')

def tile(img, ht, wt, d):
    data = []
    for h in range(0, ht, d):
        for w in range(0, wt, d):
            box = (h, w, h+d, w+d)
            a = img.crop(box)


            data.append(a)
    ret = np.array([np.array(x) for x in data])

    return ret



x = np.array([tile(Image.open(fname), 240, 320, 16) for fname in xglob])


x = np.reshape(x, (41*300, 16, 16, 3))
x = np.moveaxis(x, -1, 1)





y = np.array([tile(Image.open(fname), 480, 640, 32) for fname in yglob])

y = np.reshape(y, (41*300, 32, 32, 3))

y = np.reshape(y, (41*300, 3*32*32))
# y = np.reshape(y, (41*300, 32, 32, 3))

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=.15, random_state=6)

x = torch.tensor(X_train).type(torch.float32)
x_test = torch.tensor(X_test).type(torch.float32)

y = torch.tensor(y_train).type(torch.float32)
y_test = torch.tensor(y_test).type(torch.float32)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        v = ViT(
            image_size=16,
            patch_size=16,
            num_classes=16**2,
            dim=256,
            depth=1,
            heads=1,
            mlp_dim=256
        )
        self.transformer = v
        self.head = torch.nn.Linear(256, 3*32*32)

    def forward(self, x):
        o = self.transformer(x)
        o = self.head(o)
        return o

model = torch.load('model_ckpt')

X = torch.tensor(X_test).type(torch.float32)
d = len(X)
Y = torch.tensor(y_test).type(torch.float32).reshape((d , 32, 32, 3))


for n in [150, 175, 195, 613, 895]:
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3,
                                        figsize=(12, 6))

    index = n
    row = 5
    col = 3
    img = 0
    ax0.imshow(Y[index]/255)
    ax0.set_title("Ground Truth")
    ax1.imshow(X[index].permute(1,2,0).detach().numpy()/255)
    ax1.set_title("Input")
    ax2.imshow(model([X[index]]).detach().numpy().reshape((32,32,3))/255)
    ax2.set_title("Prediction")
    plt.show()