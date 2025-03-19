import matplotlib.pyplot as plt
import numpy as np
import strauss
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import binned_statistic_2d as bs2d
from scipy.stats import binned_statistic as bs1d
import scipy.ndimage as ndimage
import cv2
import glob
import matplotlib.pyplot as plt
import ffmpeg as ff
import wavio as wav
from strauss.sonification import Sonification
from strauss.sources import Objects
from strauss import channels
from strauss.score import Score
import numpy as np
from strauss.generator import Synthesizer, Sampler
import IPython.display as ipd
import os
from scipy.interpolate import interp1d
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator as rgi

statues = ['sphinx',
           'sekhmet',
           'persephone',
           'athena']

skips = dict(zip(statues, [0]+[2]*3))
delims = dict(zip(statues, [',']+[' ']*3))
skips = dict(zip(statues, [0]+[2]*3))
offsets = dict(zip(statues, [dict(zip(statues, [0]+[2]*3))]+[2]*3))
gids = dict(zip(statues,['1AHCrrHmG9ZCZT9VzWEEnIDMHmbzKlwM5',
                         '14r6Ym1267Vn_BPeY_Zu1XyeT_0cCkCML',
                         '1pXyV5LnjIIihF5PpagIIfhscamygDjVl',
                         '1Xp8WD7RtjEiz4HscUMLVqdSUDy0CDZrE']))
heads = dict(zip(statues,["/content/Sphynx_Head_Pointcloud_Data.txt",
                          "/content/Sekhmet_Point_Cloud_Data.txt",
                          "/content/Persephone_Point_Cloud_Data.txt",
                          "/content/Athena_Point_Cloud_Data.txt"]))

# head = 'sekhmet'



# pl = pv.Plotter(window_size=(1000, 1000))
# # points is a 3D numpy array (n_points, 3) coordinates of a sphere
# cloud = pv.PolyData(np.append(xyz, )
# cloud.plot(smooth_shading=True, color='black')

def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and
        new axes must divide old ones.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    operation = operation.lower()
    if not operation in ['sum', 'mean', 'max']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    print(range(len(new_shape)))
    for i in range(len(new_shape)):
        print(i)
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray

def get_head_data(name, fac=2, lopc=10, hipc=90):

    data = {}
    
    gdown.download(gids[name])
    points = np.genfromtxt(heads[head], delimiter=delims[head], skip_header=skips[head])
    xyz = points[::1,:-3]

    data['points'] = xyz
    
    r0 = xyz - xyz.mean(axis=0)
    zrange = np.diff(np.percentile(r0[:,-1], [0,100]))[0]
    r0[:,-1] -= -zrange * 0.33
    R = np.sqrt((r0**2).sum(axis=-1))
    polar = np.arccos(r0[:,-1]/R)
    azimuth = (np.sign(r0[:,1]) * np.arccos(r0[:,0] / np.sqrt((r0[:,:-1]**2).sum(axis=-1))))
    bdx = azimuth.argsort()[::1]
    shift = offsets[head]
    azimuth = (azimuth+np.pi-shift)%(2*np.pi) - np.pi
    polbin = np.linspace(-np.pi, np.pi, 120*fac+1)
    azibin = np.linspace(0, np.pi, 60*fac+1)

    H = bs2d(azimuth, polar, R, bins=(polbin, azibin), statistic='max')
    amin = bs1d(azimuth, R, bins=azibin, statistic=np.min)
    amax = bs1d(azimuth, R, bins=azibin, statistic=np.max)
    astd = bs1d(azimuth, R, bins=azibin, statistic=np.nanstd)

    hnormed = (np.clip(H[0]-1.5*astd[0], 0, np.inf) / (astd[0]))

    hterp = np.array(pd.DataFrame(H[0]).interpolate(limit_direction='both'))
    hterp = np.array(pd.DataFrame(hterp.T).interpolate(limit_direction='both')).T

    data = ['reliefmap']
    
    img = ndimage.gaussian_filter(hterp, sigma=2, order=0)
    img2 = ndimage.gaussian_filter(hterp, sigma=2.5, order=0)

    # Sobel Edge Detection on the X axis
    sobelx = cv2.Sobel(src=hterp.T, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3) 
    sobely = cv2.Sobel(src=hterp.T, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    sobelxy = cv2.Sobel(src=hterp.T, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3)

    clip = lambda inarr: np.clip(inarr, np.nanpercentile(inarr, lopc), np.nanpercentile(inarr, hipc))
    inquad = clip(sobelx**2+sobely**2)**2
    
    data['edgemap'] = inquad
    
    bumps = clip(np.abs(hterp-img).T/((np.abs(hterp-img2).T)))
    bumps2 =  ndimage.gaussian_filter(bumps, sigma=7, order=0)

    rearr = bin_ndarray(inquad, (inquad.shape[0]//15, inquad.shape[-1]), operation='max')
    bumps3 = bin_ndarray(bumps2, (inquad.shape[0]//15, inquad.shape[-1]), operation='mean')
    
    return data
    
def sample_points(head_data, npoints=2000):
    phi = np.linspace(0,np.pi, head_data['edgemap'].shape[0])
    theta = np.linspace(0,2*np.pi, head_data['edgemap'].shape[1])

    nquad = head_data['edgemap']/head_data['edgemap'].max()

    intp = rgi((theta, phi), nquad.T[:,::-1])
    intpr = rgi((theta, phi), head_data['reliefmap'][::1,::-1])

    points = []
    rs = []
    i = 0

    while True:
      y = np.arccos(2*np.random.random() -1 )
      x = np.random.random()*2*np.pi

      if np.random.random() < intp((x,y)):
        points.append((x,y))
        rs.append(intpr((x,y)))
        i += 1
        if i==npoints:
          break
      
    return np.array(points)
