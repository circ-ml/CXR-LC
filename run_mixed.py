"""Main testing script for the composite outcome experiment. Purpose is to determine whether using composite outcomes improves DL performance for prognosis

Usage:
  run_mixed.py <image_dir> <model_path> <data_frame> <output_file> [--TTA] [--mixedLayers=MIXEDLAYERS] [--conv=CONV] [--cont=CONT] [--cat=CAT] [--modelarch=MODELARCH] [--type=TYPE] [--target=TARGET] [--split=SPLIT] [--layers=LAYERS]
  run_mixed.py (-h | --help)
Examples:
  run_mixed.py /path/to/images /path/to/model /path/to/write/output.csv
Options:
  -h --help                     Show this screen.
  --modelarch=MODELARCH         CNN model architecture to train [default: inceptionv4]
  --type=TYPE                   Type of output [default: Discrete]
  --target=TARGET               If optional df is specified, then need to include the target variable [default: None]
  --split=SPLIT                 If split, then split on the Dataset column keeping only the Te values [default: False]
  --cont=CONT                   List of continuous variables to include in the tabular learner [default: None]
  --cat=CAT                     List of categorical variables to include in the tabular learner [default: None]
  --layers=LAYERS               Tabular layers [default: 32,32]
  --mixedLayers=MIXEDLAYERS     Mixed layers at the end of the network [default: 64,32,32,2]
  --conv=CONV                   Number of features produced by the CNN [default: 32]
  --TTA
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from docopt import docopt
import pandas as pd
import fastai
from fastai.vision import *
import pretrainedmodels
from sklearn.metrics import *
from fastai.callbacks import *
from fastai.tabular import *
import math
import time
import TabConvLearner

###TODO Add optional checkpointing (optional result file to append to, skipping loop iteration if model exists)
tfms_test = get_transforms(do_flip = False,max_warp = None,max_lighting=0,max_zoom=1.0,max_rotate=0)
tfms_train = get_transforms(do_flip = False, 
max_rotate = 5.0, max_zoom = 1.2, max_lighting=0.5,max_warp = None)

def tempTTA(learn:Learner, beta:float=0.4, scale:float=1.35, ds_type:DatasetType=DatasetType.Valid, activ:nn.Module=None, with_loss:bool=False) -> Tensors:
    "Applies TTA to predict on `ds_type` dataset."
    ##There's a batch size issue in loading a line at a time, need to resize
    #learn.model.tab_model(train_ds[0][0][0].data[0].view(1,2).cuda(),train_ds[0][0][0].data[1].view(1,1).cuda())

    #preds,y = learn.get_preds(ds_type, activ=activ)

    pbar = master_bar(range(8))
    tmp = np.zeros((len(learn.data.valid_ds),8))
    learn.model.eval()
    learn.model.tab_model.eval()
    learn.model.img_model.eval()
    
    learn.data.valid_ds.transforms = None
    preds,y = learn.get_preds(ds_type)
    #import pdb; pdb.set_trace()
    
    #ds = learn.data.valid_ds
    learn.data.valid_ds.transforms = tfms_train
    for i in pbar:
        #for j in progress_bar(range(0,len(ds)),parent=pbar):
        #    tab_res = learn.model.tab_model(ds[j][0][0].data[0].view(1,2).cuda(),ds[j][0][0].data[1].view(1,1).cuda())
        #    img_res = learn.model.img_model(ds[j][0][1].view(1,3,224,224).cuda())
        #    res = learn.model.layers(torch.cat([tab_res,img_res],dim=1))
        #    if(activ is not None):
        #        tmp[j,i] = np.array(F.softmax(res.data,dim=1)[:,1])
        #    else:
        #        tmp[j,i] = np.array(res.data)
        tmp[:,i] = learn.get_preds(ds_type)[0].data.numpy()[:,1]
    #if(activ is not None):
    #    preds = F.softmax(preds)[:,1]
    avg_preds = np.mean(tmp,axis=1)

    if beta is None: return preds[:,1].data.numpy(),avg_preds,y
    else:
        final_preds = preds[:,1].data.numpy()*beta + avg_preds*(1-beta)
        return final_preds, y

num_workers = 16
bs = 1
if __name__ == '__main__':

    arguments = docopt(__doc__)        
    ###Grab image directory
    image_dir = arguments['<image_dir>']
    
    ###Load imaging model
    mdl_path = arguments['<model_path>']

    ###set model architecture
    m = arguments['--modelarch'].lower()
    
    ###Read tabular data
    output_df = pd.read_csv(arguments['<data_frame>'])
    
    #Read target variable
    col = arguments['--target']
        
    # Split dataset
    if(arguments["--split"]!="False"):
        output_df = output_df.loc[output_df.Dataset=="Te",]
    
    # Create imagelist
    imgs = (ImageList.from_df(df=output_df,path=image_dir)
                                .split_none()
                                .label_from_df(cols=col)
                                .transform(tfms_test,size=224)
                                .databunch(num_workers = num_workers,bs=bs).normalize(imagenet_stats))
    if(m=="inceptionv4"):
        def get_model(pretrained=True, model_name = 'inceptionv4', **kwargs ): 
            if pretrained:
                arch = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
            else:
                arch = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained=None)
            return arch

        def get_cadene_model(pretrained=True, **kwargs ): 
            return fastai_inceptionv4
        custom_head = create_head(nf=2048*2, nc=37, ps=0.75, bn_final=False) 
        fastai_inceptionv4 = nn.Sequential(*list(children(get_model(model_name = 'inceptionv4'))[:-2]),custom_head) 

    ###Based on the input model, create a cnn learner object
    
    elif(m=="resnet50"):
        mdl = fastai.vision.models.resnet50
    elif(m=="resnet34"):
        mdl = fastai.vision.models.resnet34
    elif(m=="resnet16"):
        mdl = fastai.vision.models.resnet16
    elif(m=="resnet101"):
        mdl = fastai.vision.models.resnet101
    elif(m=="resnet152"):
        mdl = fastai.vision.models.resnet152
    elif(m=="densenet121"):
        mdl = fastai.vision.models.densenet121
    elif(m=="densenet169"):
        mdl = fastai.vision.models.densenet169
    elif(m=="age"):
        mdl=fastai.vision.models.resnet34
    else:
        print("Sorry, model: " + m + " is not yet supported... coming soon!")
        quit()
    
    
    
    if(m=='inceptionv4'):
        img_learn = cnn_learner(imgs, get_cadene_model, metrics=accuracy)
    else:
        img_learn = cnn_learner(imgs, mdl, metrics=accuracy)

    if(m=="age"):
        numFeatures = 16
        img_learn.model[1] = nn.Sequential(*img_learn.model[1][:-5],nn.Linear(1024,512,bias=True),nn.ReLU(inplace=True),nn.BatchNorm1d(512),nn.Dropout(p=0.5),
                             nn.Linear(512,numFeatures,bias=True),nn.ReLU(inplace=True),nn.BatchNorm1d(numFeatures),
                             nn.Linear(numFeatures,1,bias=True)).cuda()
     
    
    ###Load tabular learner
    cont = []
    cat = []
    if(arguments['--cont']!="None"):
        cont = arguments['--cont'].split(sep=',')
    if(arguments['--cat']!="None"):
        cat = arguments['--cat'].split(sep=',')
    
    procs = [Categorify]
    data_tab = (TabularList.from_df(output_df, cat, cont, procs=procs)
            .split_none()
            #.split_by_rand_pct(valid_pct = 0.2)
            .label_from_df(cols=col)
            .databunch())
    tab_layers = arguments['--layers'].split(sep=',')
    tab_layers = [int(f) for f in tab_layers]
    learn_tab = tabular_learner(data_tab, layers=tab_layers, ps=[0.25], emb_drop=0.05, metrics=accuracy)
    

    train_ds = TabConvLearner.TabConvDataset(data_tab.train_ds.x, imgs.train_ds.x, data_tab.train_ds.y,transforms=tfms_test,size=224)
    valid_ds = TabConvLearner.TabConvDataset(data_tab.train_ds.x, imgs.train_ds.x, data_tab.train_ds.y,transforms=tfms_test,size=224)

    train_dl = DataLoader(train_ds, bs,shuffle=False)
    valid_dl = DataLoader(valid_ds, 2 * bs,shuffle=False)

    data = DataBunch(train_dl, valid_dl)
    
    #Chop off last layer of tabular model
    learn_tab.model.layers = learn_tab.model.layers[:-1]
    
    #Should be able to compute this
    n_lin_conv = int(arguments['--conv'])

    img_out = 3072
    if("resnet" in m):
        img_out = 1024
        
        
    img_learn.model[-1] = nn.Sequential(*img_learn.model[-1][:-5], nn.Linear(img_out, n_lin_conv, bias=True), nn.ReLU(inplace=True))
    
    
    ps_final = 0.25
    n_lin_tab = tab_layers[len(tab_layers)-1]
    
    lin_layers = arguments['--mixedLayers'].split(sep=',')
    lin_layers = [int(f) for f in lin_layers]
    if(lin_layers[0]!=n_lin_conv+n_lin_tab):
        print("Mismatch, number of layers in combined input is: " + str(lin_layers[0]) + ", Conv features: " + str(n_lin_conv) + ", Tabular features: " + str(n_lin_tab))
        sys.exit(0)
        
    
    ps = np.repeat(ps_final,len(lin_layers)-1)
    
    flatten_model(img_learn.layer_groups[-1])
    
    model = TabConvLearner.TabConvModel(learn_tab.model, img_learn.model, lin_layers, ps)


    layer_groups = [nn.Sequential(*flatten_model(img_learn.layer_groups[0])),
                    nn.Sequential(*flatten_model(img_learn.layer_groups[1])+
                    #nn.Sequential(*(flatten_model(img_learn.layer_groups[2]) +
                                    flatten_model(model.tab_model) +
                                    flatten_model(model.layers))
                   ]

    # combined learner
    learn = Learner(data, model,
                    layer_groups=layer_groups
                   )
    learn.model_dir = "."
    learn.load(arguments['<model_path>'])
    if(arguments['--TTA']):

        if(arguments['--type'].lower()=="discrete"):
            preds,y = tempTTA(learn,ds_type = DatasetType.Fix,activ=nn.Softmax())
        else:
            preds,y = tempTTA(learn,ds_type = DatasetType.Fix)
    else:
        preds,y = learn.get_preds(DatasetType.Fix)
        preds = preds.data.numpy()[:,1]

    ###output predictions as column with model name
    output_df['CXRLC_Probability'] = np.exp(preds*16.599 - 7.266) / (1 + np.exp(16.599*preds - 7.266))
    output_df.to_csv(arguments['<output_file>'],index=False)
