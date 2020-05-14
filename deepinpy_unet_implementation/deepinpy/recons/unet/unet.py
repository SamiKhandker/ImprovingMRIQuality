#!/usr/bin/env python

from deepinpy.forwards import MultiChannelMRI
from deepinpy.models import UNet
from deepinpy.recons import Recon

class UNetRecon(Recon):

    def __init__(self, hparams):
        super(UNetRecon, self).__init__(hparams)
        if self.hparams.network == 'UNet':
            self.network = UNet(batch_norm=self.hparams.batch_norm , l2lam = self.hparams.l2lam_init)

    def batch(self, data):

        maps = data['maps']
        masks = data['masks']
        inp = data['out']

        self.A = MultiChannelMRI(maps, masks, l2lam=self.hparams.l2lam_init,  img_shape=data['imgs'].shape, use_sigpy=self.hparams.use_sigpy, noncart=self.hparams.noncart)
        self.x_adj = self.A.adjoint(inp)

    def forward(self, y):
        return self.network(self.x_adj)

    def get_metadata(self):
        return {}