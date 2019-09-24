import os
import re
import glob
import errno
import numpy as np
import pandas as pd

from shutil import copyfile



IMAGES_PATH = '/home/kwhuang/kdeimages'
SUMMARY_DIR = 'candidates'


def create_dir(dir_name: str):
    """ Create directory with a name 'dir_name' """
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def dist2(x: np.ndarray, y: np.ndarray, x0: float, y0: float) -> np.ndarray:
    """ Calculate the square of distance between a numpy array and (x0, y0).
    : x : x coordinates
    : y : y coordinates
    : x0 : x center of the reference point
    : y0 : y center of the reference point
    : return : number of pixels of the inner aperture
    """
    dx = x - x0
    dy = y - y0
    return  dx ** 2 + dy ** 2





if __name__ == '__main__':

    create_dir(SUMMARY_DIR)

    images_list = []
    images_paths = glob.glob(IMAGES_PATH + '/*')
    for path in images_paths:
        images_files = glob.glob(path + '/*')
        images_list += images_files

    ras, decs, sigs, s1_degs, s1_pcs, dwarfs, im_names = [], [], [], [], [], [], []
    for im in images_list:
        imagename = im.split('/')[-1]
        dwarf = imagename.split('=')[0]

        s = imagename.split('=')[1]
        ra = float(re.search('-ra(.*)-dec', s).group(1))
        dec = float(re.search('-dec(.*)-sig', s).group(1))
        sig = float(re.search('-sigp(.*).jpg', s).group(1))

        kernels = re.search('-gc(.*)-poisson', s).group(1).split('s')
        s1_pc, s1_deg = int(kernels[0]), float(kernels[1])

        ras.append(ra)
        decs.append(dec)
        sigs.append(sig)
        dwarfs.append(dwarf)
        s1_pcs.append(s1_pc)
        s1_degs.append(s1_deg)
        im_names.append(imagename)


    d = {}
    d['ra'] = np.array(ras)
    d['dec'] = np.array(decs)
    d['sig'] = np.array(sigs)
    d['path'] = np.array(images_list)
    d['dwarf'] = np.array(dwarfs)
    d['s1_pc'] = np.array(s1_pcs)
    d['s1_deg'] = np.array(s1_degs)
    d['im_name'] = np.array(im_names)



    df = pd.DataFrame(d).sort_values(by=['sig'], ascending=False)
    df = df.reset_index(drop=True)


    for idx, row in df.iterrows():
        dist = np.sqrt(dist2(df['ra'][:idx], df['dec'][:idx], row['ra'], row['dec']))
        mask = (0 < dist) & (dist < 2. * row['s1_deg'])
        mask = mask & (row['dwarf']==df['dwarf'][:idx])
        if np.sum(mask) > 0:    # skip same target
            continue
        copyfile(row['path'], SUMMARY_DIR + '/' + row['im_name'])
