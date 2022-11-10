import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

PATH = 'Temperature Analysis/Whole Plane Analysis'
PPMAP_PATH = 'PPMAP_Results'
T_CRIT = [12.5, 14.4, 16.7, 19.3, 22.4, 25.9]
resolution = 0.001667

bins = np.logspace(np.log10(1e20),
                   np.log10(2e23), 
                   100)

N_min = '10'
try: 
    os.makedirs(f'{PATH}/{N_min}') 
except OSError as error: 
    pass


np.savetxt(f'{PATH}/{N_min}/bins.csv', bins)
df_filter = pd.read_csv(f'{PATH}/Filters/{N_min}.csv')
df_filter['Latitude'] = ((df_filter[['Latitude']].div(resolution)).round().mul(resolution))
df_filter['Longitude'] = ((df_filter[['Longitude']].div(resolution)).round().mul(resolution))
df_filter = df_filter.drop_duplicates(['Latitude','Longitude'])

df = pd.DataFrame()
for folder in tqdm(os.listdir(PPMAP_PATH)):
    tilename = folder[:4]
    hdu = fits.open(f'{PPMAP_PATH}/{folder}/{tilename}_diffcdens.fits')[0]
    wcs = WCS(hdu.header)
    images = hdu.data
    df_i = pd.DataFrame()
    for i,image in enumerate(images):
        df_ii = pd.DataFrame()
        shape = np.shape(image)
        x = np.arange(0, shape[0])
        y = np.arange(0, shape[1])
        xx, yy = np.meshgrid(x, y)
        coords_grid = pixel_to_skycoord(xx,yy, wcs)
        lat = coords_grid.l.degree
        lon = coords_grid.b.degree
        df_ii['Column Density'] = image.flatten()*10**20
        df_ii['Latitude'] = lat.flatten()
        df_ii['Longitude'] = lon.flatten()
        df_ii['Latitude'] = ((df_ii[['Latitude']].div(resolution)).round().mul(resolution))
        df_ii['Longitude'] = ((df_ii[['Longitude']].div(resolution)).round().mul(resolution))
        df_ii['T'] = hdu.header[f'TEMP{i+1:02}']
        df_ii.dropna()
        df_ii = df_ii[df_ii['Column Density']>0]
        df_ii = df_ii.merge(df_filter, how='right', on = ['Latitude', 'Longitude']).dropna()
        df_i = df_i.append(df_ii)
    df = df.append(df_i)

for T in tqdm(T_CRIT):
	df_t = pd.DataFrame()
	T_round = np.round(T)
	plt.figure()
	df_t['n_all'],bins_all,_ = plt.hist(df[['Latitude','Longitude','Column Density']].groupby(['Latitude','Longitude']).sum()['Column Density'], 
	         log=True, 
	         bins = bins,
	         histtype = u'step',
	         color='k',
	         label= 'BLACK: ALL T')
	df_t['n_hot'],bins_all,_ = plt.hist(df[df['T']>= T][['Latitude','Longitude','Column Density']].groupby(['Latitude','Longitude']).sum()['Column Density'], 
	         log=True, 
	         bins = bins,
	         histtype = u'step',
	         color='b',
	         label= fr'$\mathrm{{BLUE:}} T \geq {T}$')
	df_t['n_cold'],bins_all,_ = plt.hist(df[df['T']<T][['Latitude','Longitude','Column Density']].groupby(['Latitude','Longitude']).sum()['Column Density'], 
	         log=True, 
	         bins = bins,
	         histtype = u'step',
	         color='r',
	         label= fr'$\mathrm{{RED:}} T < {T}$')
	plt.xscale('log')
	plt.xlabel(r'$N(H_2) [cm^{-2}]$')
	plt.ylabel(r'$dN/d\logN(H_2)[\mathrm{per bin}]$')
	plt.title(f'{N_min}x10^21 N_H2/cm^-2')
	plt.legend()
	plt.ylim(1e1)
	plt.xlim(3e20)
	plt.savefig(f'{PATH}/{N_min}/{int(T_round):02}.png')
	df_t.to_csv(f'{PATH}/{N_min}/{int(T_round):02}_hist.csv', index=False)