import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal


class ConvBlock(nn.Module):
    """
    Reference:
    From here.
    https://github.com/voxelmorph/voxelmorph/blob/master/voxelmorph/torch/networks.py
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, "Conv%dd" % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out


class Unet(nn.Module):
    """
    Reference:
    Modified from here.
    https://github.com/voxelmorph/voxelmorph/blob/master/voxelmorph/torch/networks.py

    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__()
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], (
            "ndims should be one of 1, 2, or 3. found: %d" % ndims
        )

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError(
                    "must provide unet nb_levels if nb_features is an integer"
                )
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(
                int
            )
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError("cannot use nb_levels if nb_features is not an integer")
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # configure encoder (down-sampling path)
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[: len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf) :]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

    def forward(self, x):
        # get encoder activations
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))

        # conv, upsample, concatenate series
        x = x_enc.pop()
        for layer in self.uparm:
            x = layer(x)
            x = self.upsample(x)
            _x = x_enc.pop()
            try:
                x = torch.cat([x, _x], dim=1)
            except RuntimeError:
                x3d = list(x.shape)[2]
                _x3d = list(_x.shape)[2]
                if x3d > _x3d:
                    pd = (0, 0, x3d - _x3d, 0)
                    _x = F.pad(_x, pd, mode="constant", value=0)
                elif x3d < _x3d:
                    pd = (0, 0, _x3d - x3d, 0)
                    x = F.pad(x, pd, mode="constant", value=0)
                x = torch.cat([x, _x], dim=1)
        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)

        return x


class SpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to perform a grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """

    def __init__(self, size, mode="bilinear"):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer, self).__init__()

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer("grid", grid)

        self.mode = mode

    def forward(self, src, flow):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        try:
            new_locs = self.grid + flow
        except RuntimeError:
            new_locs = self.grid[:, :, : list(flow.shape)[2], :]
        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode, align_corners=True)


class FramePredNet(nn.Module):
    """ "" implementation of voxelmorph."""

    def __init__(self, vol_size, enc_nf, dec_nf, full_size=True):
        """
        :param vol_size: volume size of the atlas
        :param enc_nf: the number of features maps for encoding stages
        :param dec_nf: the number of features maps for decoding stages
        :param full_size: boolean value full amount of decoding layers
        """
        super(FramePredNet, self).__init__()

        dim = len(vol_size)
        self.unet_model = Unet(vol_size, [enc_nf, dec_nf])

        # One conv to get the flow field
        conv_fn = getattr(nn, "Conv%dd" % dim)
        self.flow = conv_fn(dec_nf[-1], dim, kernel_size=3, padding=1)

        # Make flow weights + bias small. Not sure this is necessary.
        normal_dist = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(normal_dist.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        self.spatial_transform = SpatialTransformer(vol_size)

    def forward(self, src, tgt):
        """
        Pass input x through forward once
            :param src: moving image that we want to shift
            :param tgt: fixed image that we want to shift to
        """
        x = torch.cat([src, tgt], dim=1)
        x = self.unet_model(x)
        flow = self.flow(x)
        y = self.spatial_transform(src, flow)

        return y, flow


def build_model(
    vol_shape=None, pretrained_model=None, nb_enc_features=None, nb_dec_features=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if vol_shape is None:
        # pretrained model with input volume shape
        vol_shape = (672, 1280)
    if pretrained_model is None and vol_shape == (672, 1280):
        pretrained_model = str(
            Path(__file__).parent / "weights" / "deformation_latest.pt"
        )
    if nb_enc_features is None:
        nb_enc_features = [32, 32, 32, 32]
    if nb_dec_features is None:
        nb_dec_features = [32, 32, 32, 32, 32, 16]
    fpnet = FramePredNet(vol_shape, nb_enc_features, nb_dec_features)
    fpnet.to(device)
    if pretrained_model is not None and os.path.exists(pretrained_model):
        fpnet.load_state_dict(torch.load(pretrained_model, map_location=device))
        return fpnet
    else:
        return None
