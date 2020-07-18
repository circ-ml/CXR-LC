from fastai import *
from fastai.tabular import *
from fastai.vision import *
import fastai.data_block

# transformations for image augmentation
tfms = get_transforms(do_flip=True,
                      flip_vert=True,
                      max_rotate=15,
                      max_zoom=1.05,
                      max_warp=0,
                      max_lighting=0,
)

class TabConvDataset(Dataset):
    """A Dataset of combined tabular data, image names, and targets."""
    def __init__(self, x_tab, x_img, y,transforms=None,size=None):
        self.x_tab, self.x_img, self.y = x_tab, x_img, y
        self.is_empty = False
        self.set = False
        self.transforms = transforms
        self.size = size
        if isinstance(y,fastai.data_block.MultiCategoryList):
            self.c = 2
        else:
            self.c = len(np.unique(np.asarray(y,dtype='int64')))

    def __len__(self): return len(self.y)

    def __getitem__(self, i,curr_trans = None):
        if(curr_trans is None and self.transforms is not None):
            curr_trans = self.transforms
        if(curr_trans is not None):
            img = self.x_img[i]
            if(self.size is not None):
                img = img.apply_tfms(curr_trans[0],size=self.size).px
            else:
                img = img.apply_tfms(curr_trans[0]).px
        else:
            img = self.x_img[i]
        ###Can apply transformation to x_img if this is the training set TODO
        if(self.set):
            return (self.x_tab[i],img), (self.y[i],self.weights[i])
        else:
            return (self.x_tab[i],img),self.y[i]

    def setWeights(self,y2):
        self.set = True
        self.weights = y2
class TabConvModel(nn.Module):
    """A combined neural network using the convnet and tabular model"""
    def __init__(self, tab_model, img_model, layers, drops):
        super().__init__()
        self.tab_model = tab_model
        self.img_model = img_model
        lst_layers = []

        activs = [nn.ReLU(inplace=True),] * (len(layers) - 2) + [None]

        for n_in, n_out, p, actn in zip(layers[:-1], layers[1:], drops, activs):
            lst_layers += bn_drop_lin(n_in, n_out, p=p, actn=actn)

        self.layers = nn.Sequential(*lst_layers)

    def forward(self, *x):
        x_tab = self.tab_model(*x[0])
        x_img = self.img_model(x[1])

        x = torch.cat([x_tab, x_img], dim=1)
        return self.layers(x)