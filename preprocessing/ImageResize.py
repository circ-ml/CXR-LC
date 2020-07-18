""" Resizing images to 224 x 224 using bilinear interpolation

Usage:
  ImageResize.py <base_dir> <out_dir>
  ImageResize.py (-h | --help)
Examples:
  python ImageResize /path/to/folder/large/size/images/ /path/to/output/directory/ 
Options:
  -h --help                          Show this screen.

"""



from fastai.vision import *
from docopt import docopt
if __name__ == '__main__':

    #Grab arguments from docopt
    arguments = docopt(__doc__)
    
    #High Res images
    path_hr = Path(arguments['<base_dir>'])

    #Low res output
    path_lr = Path(arguments['<out_dir>'])
    
    #Get list of images using fastai
    il = ImageList.from_folder(path_hr)

    #Use PIL to perform bilinear interpolation to 224 x 224
    def resize_one(fn, i, path, size):
        dest = path/fn.relative_to(path_hr)
        dest.parent.mkdir(parents=True, exist_ok=True)
        img = PIL.Image.open(fn)
        img = img.resize([224,224], resample=PIL.Image.BILINEAR).convert('RGB')
        img.save(dest, quality=75)

    #Resize and output to output directory
    sets = [(path_lr, 224)]
    for p,size in sets:
        if not p.exists(): 
            print(f"resizing to {size} into {p}")
            parallel(partial(resize_one, path=p, size=size), il.items)