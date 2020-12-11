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

model = Model()

crit = torch.nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=4e-4)

model.train()
epoch = 1500*10
epocharr = []
lossarr = []
testlossarr = []
min_test_loss = 1000
min_loss = 1000
for epoch in range(epoch):
    opt.zero_grad()
    # Forward pass
    y_pred = model(x)
    # Compute Loss
    loss = crit(y_pred.squeeze(), y)
    test_loss = crit(model(x_test), y_test)

    if loss.item() < min_loss and test_loss.item() < min_test_loss:
        min_loss = loss.item()
        min_test_loss = test_loss.item()
        torch.save(model, f'./model_ckpt')

    print(f'Epoch {epoch}: train loss: {format(loss.item(), ".3f")} || test loss: {format(test_loss.item(), ".3f")}')
    epocharr.append(epoch)

    lossarr.append(loss.item())
    testlossarr.append(test_loss.item())
    # Backward pass
    loss.backward()
    opt.step()


torch.save(model, "./mode_fix")

plt.plot(epocharr, lossarr)
plt.plot(epocharr, testlossarr)
plt.show()