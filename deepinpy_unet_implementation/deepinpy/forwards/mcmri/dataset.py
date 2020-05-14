#!/usr/bin/env python

import torch.utils.data
import numpy as np
import scipy.fftpack
import h5py
import pathlib

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import deepinpy.utils.complex as cp
from deepinpy.utils.utils import fftmod, fftshift, fft2uc, ifft2uc

'''
Simulates data and creates Dataset objects for pytorch
'''

class MultiChannelMRIDataset(torch.utils.data.Dataset):
    ''' Multichannel MRI data set '''

    def __init__(self, data_file, stdev=.01, num_data_sets=None, adjoint=True, preload=False, id=None, clear_cache=False, cache_data=False, gen_masks=False, scale_data=False, fully_sampled=False, data_idx=None, inverse_crime=True, noncart=False):

        self.data_file = data_file
        self.stdev = stdev
        self.adjoint = adjoint
        self.preload = preload
        self.id = id
        self.cache_data = cache_data
        self.gen_masks = gen_masks
        self.max_data_sets = self._len()
        self.fully_sampled = fully_sampled
        self.data_idx = data_idx
        self.scale_data = scale_data
        self.inverse_crime = inverse_crime
        self.noncart = noncart

        if self.data_idx is not None:
            self.num_data_sets = 1
        else:
            if num_data_sets is None:
                self.num_data_sets = self.max_data_sets
            else:
                self.num_data_sets = int(np.minimum(num_data_sets, self.max_data_sets))

        if self.cache_data and clear_cache:
            print('clearing cache')
            for p in pathlib.Path('.').glob('cache/{}_*_{}'.format(self.id, self.data_file)):
                pathlib.Path.unlink(p)

    def _len(self):
        with h5py.File(self.data_file, 'r') as F:
            max_data_sets = F['imgs'].shape[0]
        return max_data_sets

    def __len__(self):
        return self.num_data_sets

    def __getitem__(self, idx):
        if self.data_idx is not None:
            idx = self.data_idx

        if self.cache_data and self.id:
            data_file = 'cache/{}_{}_{}'.format(self.id, idx, self.data_file)
            try:
                imgs, maps, masks, out = load_data_cached(data_file)
            except:
                imgs, maps, masks, out = self._load_data(idx)
                save_data_cached(data_file, imgs, maps, masks, out)
        else:
                imgs, maps, masks, out = self._load_data(idx)

        data = {
                'imgs': cp.c2r(imgs.squeeze()).astype(np.float32),
                'maps': cp.c2r(maps.squeeze()).astype(np.float32),
                'masks': masks.squeeze().astype(np.float32),
                'out': cp.c2r(out.squeeze()).astype(np.float32)
                }

        return idx, data

    def _load_data(self, idx):
        if self.inverse_crime:
            #imgs, maps, masks = load_data_legacy(idx, self.data_file, self.gen_masks)
            imgs, maps, masks = load_data(idx, self.data_file, self.gen_masks)
        else:
            imgs, maps, masks, ksp = load_data_ksp(idx, self.data_file, self.gen_masks)
        if self.scale_data:
            ## FIXME: batch mode
            assert not self.scale_data, 'SEE FIXME'
            sc = np.percentile(abs(imgs), 99, axis=(-1, -2))
            imgs = imgs / sc
            ksp = ksp / sc

        if self.fully_sampled:
            masks = np.ones(masks.shape)

        if self.inverse_crime:
            assert not self.noncart, 'FIXME: forward sim of NUFFT'
            out = self._sim_data(imgs, maps, masks)
        else:
            out = self._sim_data(imgs, maps, masks, ksp)

        if not self.noncart:
            maps = fftmod(maps)
        return imgs, maps, masks, out

    def _sim_data(self, imgs, maps, masks, ksp=None):

        # N, nc, nx, ny
        if self.noncart:
            assert ksp is not None, 'FIXME: NUFFT forward sim'
            noise = np.random.randn(*ksp.shape) + 1j * np.random.randn(*ksp.shape)
        else:
            noise = np.random.randn(*maps.shape) + 1j * np.random.randn(*maps.shape)

        if self.inverse_crime and ksp is None:
            out = masks[:,None,:,:] * (fft2uc(imgs[:,None,:,:] * maps) + 1 / np.sqrt(2) * self.stdev * noise)
        else:
            if self.noncart:
                out = ksp + 1 / np.sqrt(2) * self.stdev * noise
            else:
                out = masks[:,None,:,:] * (ksp + 1 / np.sqrt(2) * self.stdev * noise)

        if self.adjoint:
            assert not self.noncart, 'FIXME: support NUFFT sim'
            out = np.sum(np.conj(maps) * ifft2uc(out), axis=1).squeeze()
        else:
            if not self.noncart:
                out = fftmod(out)

        return out


def load_data(idx, data_file, gen_masks=False):
    with h5py.File(data_file, 'r') as F:
        imgs = np.array(F['imgs'][idx,...], dtype=np.complex)
        maps = np.array(F['maps'][idx,...], dtype=np.complex)
        masks = np.array(F['masks'][idx,...], dtype=np.float)

    # special case for batch_size=1
    if len(masks.shape) == 2:
        imgs, maps, masks = imgs[None,...], maps[None,...], masks[None,...]
    return imgs, maps, masks

def load_data_ksp(idx, data_file, gen_masks=False):
    with h5py.File(data_file, 'r') as F:
        imgs = np.array(F['imgs'][idx,...], dtype=np.complex)
        maps = np.array(F['maps'][idx,...], dtype=np.complex)
        ksp = np.array(F['ksp'][idx,...], dtype=np.complex)
        masks = np.array(F['masks'][idx,...], dtype=np.float)

    # special case for batch_size=1
    if len(masks.shape) == 2:
        imgs, maps, masks, ksp = imgs[None,...], maps[None,...], masks[None,...], ksp[None,...]
    return imgs, maps, masks, ksp


def load_data_cached(data_file):
    with h5py.File(data_file, 'r') as F:
        imgs = np.array(F['imgs'], dtype=np.complex)
        maps = np.array(F['maps'], dtype=np.complex)
        masks = np.array(F['masks'], dtype=np.float)
        out = np.array(F['out'], dtype=np.complex)
    return imgs, maps, masks, out


def save_data_cached(data_file, imgs, maps, masks, out):
    with h5py.File(data_file, 'w') as F:
        F.create_dataset('imgs', data=imgs)
        F.create_dataset('maps', data=maps)
        F.create_dataset('masks', data=masks)
        F.create_dataset('out', data=out)
