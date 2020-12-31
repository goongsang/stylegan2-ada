# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# """Generate images using pretrained network pickle."""

# import argparse
# import os
# import pickle
# import re

# import numpy as np
# import PIL.Image

# import dnnlib
# import dnnlib.tflib as tflib

# from libObj import libObj

# #----------------------------------------------------------------------------

# def generate_images(network_pkl, seeds, truncation_psi, outdir, class_idx, dlatents_npz):
#     tflib.init_tf()
#     print('Loading networks from "%s"...' % network_pkl)
#     with dnnlib.util.open_url(network_pkl) as fp:
#         _G, _D, Gs = pickle.load(fp)

#     os.makedirs(outdir, exist_ok=True)

#     # Render images for a given dlatent vector.
#     if dlatents_npz is not None:
#         print(f'Generating images from dlatents file "{dlatents_npz}"')
#         dlatents = np.load(dlatents_npz)['dlatents']
#         assert dlatents.shape[1:] == (18, 512) # [N, 18, 512]
#         imgs = Gs.components.synthesis.run(dlatents, output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True))
#         for i, img in enumerate(imgs):
#             fname = f'{outdir}/dlatent{i:02d}.png'
#             print (f'Saved {fname}')
#             PIL.Image.fromarray(img, 'RGB').save(fname)
#         return

#     # Render images for dlatents initialized from random seeds.
#     Gs_kwargs = {
#         'output_transform': dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
#         'randomize_noise': False
#     }
#     if truncation_psi is not None:
#         Gs_kwargs['truncation_psi'] = truncation_psi

#     noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
#     label = np.zeros([1] + Gs.input_shapes[1][1:])
#     if class_idx is not None:
#         label[:, class_idx] = 1

#     for seed_idx, seed in enumerate(seeds):
#         print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
#         rnd = np.random.RandomState(seed)
#         z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
#         tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
#         images = Gs.run(z, label, **Gs_kwargs) # [minibatch, height, width, channel]
#         PIL.Image.fromarray(images[0], 'RGB').save(f'{outdir}/seed{seed:04d}.png')

# #----------------------------------------------------------------------------

# def _parse_num_range(s):
#     '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

#     range_re = re.compile(r'^(\d+)-(\d+)$')
#     m = range_re.match(s)
#     if m:
#         return list(range(int(m.group(1)), int(m.group(2))+1))
#     vals = s.split(',')
#     return [int(x) for x in vals]

# #----------------------------------------------------------------------------

# _examples = '''examples:

#   # Generate curated MetFaces images without truncation (Fig.10 left)
#   python %(prog)s --outdir=out --trunc=1 --seeds=85,265,297,849 \\
#       --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metfaces.pkl

#   # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
#   python %(prog)s --outdir=out --trunc=0.7 --seeds=600-605 \\
#       --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metfaces.pkl

#   # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
#   python %(prog)s --outdir=out --trunc=1 --seeds=0-35 --class=1 \\
#       --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/cifar10.pkl

#   # Render image from projected latent vector
#   python %(prog)s --outdir=out --dlatents=out/dlatents.npz \\
#       --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/ffhq.pkl
# '''

# #----------------------------------------------------------------------------

# def main():
#     parser = argparse.ArgumentParser(
#         description='Generate images using pretrained network pickle.',
#         epilog=_examples,
#         formatter_class=argparse.RawDescriptionHelpFormatter
#     )

#     parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
#     g = parser.add_mutually_exclusive_group(required=True)
#     g.add_argument('--seeds', type=_parse_num_range, help='List of random seeds')
#     g.add_argument('--dlatents', dest='dlatents_npz', help='Generate images for saved dlatents')
#     parser.add_argument('--trunc', dest='truncation_psi', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
#     parser.add_argument('--class', dest='class_idx', type=int, help='Class label (default: unconditional)')
#     parser.add_argument('--outdir', help='Where to save the output images', required=True, metavar='DIR')

#     args = parser.parse_args()
#     generate_images(**vars(args))

# #----------------------------------------------------------------------------

# if __name__ == "__main__":
#     main()

# #----------------------------------------------------------------------------

# https://github.com/kevinzakka/spatial-transformer-network/blob/master/stn/transformer.py
def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W,)
    - y: flattened tensor of shape (B*H*W,)
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)

def bilinear_sampler(img, x, y):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H - 1, tf.int32)
    max_x = tf.cast(W - 1, tf.int32)
    zero = tf.zeros([], dtype=tf.int32)

    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    # x = 0.5 * ((x + 1.0) * tf.cast(max_x-1, tf.float32))
    # y = 0.5 * ((y + 1.0) * tf.cast(max_y-1, tf.float32))
    x = x * tf.cast(max_x-1, tf.float32)
    y = y * tf.cast(max_y-1, tf.float32)

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), tf.int32)
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), tf.int32)
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, tf.float32)
    x1 = tf.cast(x1, tf.float32)
    y0 = tf.cast(y0, tf.float32)
    y1 = tf.cast(y1, tf.float32)

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

    return out



import argparse
import os
import pickle
import re

import numpy as np
import PIL.Image

import dnnlib
import dnnlib.tflib as tflib

from importlib import reload
import libObj
reload(libObj)
import tensorflow as tf

tflib.init_tf()

# network_pkl = "/home/jseo/nvidia/stylegan2-ada/geom-train/00002-rasters_tfr-auto1-gamma100/network-snapshot-001200.pkl"
# network_pkl = "/home/jseo/Downloads/network-snapshot-002048-g25.pkl"
# network_pkl = "/home/jseo/Downloads/1661853/00000-rasters_tfr-res256-stylegan2-noaug/network-snapshot-001228.pkl"
# network_pkl = "/home/jseo/Downloads/1660968/00000-rasters_tfr-res256-auto4-gamma25/network-snapshot-007577.pkl"
# network_pkl = "/home/jseo/Downloads/1661853-2/00000-rasters_tfr-res256-stylegan2-noaug/network-snapshot-006348.pkl"
# network_pkl = "/home/jseo/Downloads/1660972/00000-rasters_tfr-res256-auto4-gamma2/network-snapshot-014745.pkl"
# network_pkl = "/home/jseo/Downloads/1660968/00000-rasters_tfr-res256-auto4-gamma25/network-snapshot-014950.pkl"
network_pkl = "/home/jseo/Downloads/1660969/00000-rasters_tfr-res256-auto4-gamma10/network-snapshot-014131.pkl"
# network_pkl = "/home/jseo/Downloads/network-snapshot-004096.pkl"
# network_pkl = "/home/jseo/Downloads/network-snapshot-009421.pkl"
# network_pkl = "/home/jseo/Downloads/network-snapshot-002048.pkl"
# network_pkl = "/home/jseo/Downloads/network-snapshot-001024.pkl"
truncation_psi = None
class_idx = None
meanObj = libObj.libObj("/media/jseo/BigData/nvidia/stylegan3d/data/triplegangers_v1/mean.obj")
meanAndScale = np.load("/media/jseo/BigData/nvidia/stylegan3d/data/triplegangers_v1/rasters_tfr/mean_and_scale.npy", allow_pickle=True).item()
deltaScale = 1.0 / meanAndScale['deltaScale']
imageMean = tf.convert_to_tensor(meanAndScale['mean'], dtype=tf.float32)

print('Loading networks from "%s"...' % network_pkl)
with dnnlib.util.open_url(network_pkl) as fp:
    _G, _D, Gs = pickle.load(fp)

# Render images for a given dlatent vector.
# if dlatents_npz is not None:
#     print(f'Generating images from dlatents file "{dlatents_npz}"')
#     dlatents = np.load(dlatents_npz)['dlatents']
#     assert dlatents.shape[1:] == (18, 512) # [N, 18, 512]
#     imgs = Gs.components.synthesis.run(dlatents, output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True))
#     for i, img in enumerate(imgs):
#         fname = f'{outdir}/dlatent{i:02d}.png'
#         print (f'Saved {fname}')
#         PIL.Image.fromarray(img, 'RGB').save(fname)
#     return

# Render images for dlatents initialized from random seeds.
Gs_kwargs = {
    # 'output_transform': dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
    'randomize_noise': False
}
if truncation_psi is not None:
    Gs_kwargs['truncation_psi'] = truncation_psi

noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
label = np.zeros([1] + Gs.input_shapes[1][1:])
if class_idx is not None:
    label[:, class_idx] = 1

nPnts = meanObj.pnts.shape[0]
nUVs = meanObj.uvs.shape[0]
uvIdToVtxId = tf.convert_to_tensor(meanObj.uvIdToVtxId.reshape(nPnts, 1), dtype=tf.int32)
u = np.repeat(meanObj.uvs[:,0], b).reshape(b, 1, nUVs)
v = np.repeat(meanObj.uvs[:,1], b).reshape(b, 1, nUVs)
x = tf.convert_to_tensor(u, dtype=tf.float32)
y = tf.convert_to_tensor(v, dtype=tf.float32)

# def imageToMesh(image):
#     posImages = tf.transpose(image, [0, 2, 3, 1]) * deltaScale + imageMean
#     # b = posImages.shape[0]
#     # uvPnts = tf.reshape(bilinear_sampler(posImages, x, y), (b, nUVs, 3))
#     uvPnts = tf.reshape(bilinear_sampler(posImages, x, y), (1, nUVs, 3))

#     # pnts_list = []
#     # for i in range(b):
#     #     pnts_list.append(tf.gather_nd(uvPnts[i], uvIdToVtxId))
#     # pnts = tf.stack(pnts_list)
#     pnts = tf.gather_nd(uvPnts[0], uvIdToVtxId)
#     return pnts

# img2pnts = tf.function(imageToMesh)

h, w, c = imageMean.shape
uvPnts = tf.reshape(bilinear_sampler(tf.reshape(imageMean, (1,h,w,c)), x, y), (1, nUVs, 3))
pnts = tf.gather_nd(uvPnts[0], uvIdToVtxId)
meanPnts = tf.get_default_session().run(pnts)

import time
seeds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
for seed_idx, seed in enumerate(seeds):
    print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
    t = time.time()

    rnd = np.random.RandomState(seed)
    z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
    tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
    image = Gs.run(z, label, **Gs_kwargs) # [minibatch, height, width, channel]
    # PIL.Image.fromarray(images[0], 'RGB').save(f'{outdir}/seed{seed:04d}.png')
    
    with tf.device("/gpu:0"):
        posImages = tf.transpose(image, [0, 2, 3, 1]) * deltaScale
        b = posImages.shape[0]
        uvPnts = tf.reshape(bilinear_sampler(posImages, x, y), (b, nUVs, 3))
        # pnts_list = []
        # for i in range(b):
        #     pnts_list.append(tf.gather_nd(uvPnts[i], uvIdToVtxId))
        # pnts = tf.stack(pnts_list)
        pnts = tf.gather_nd(uvPnts[0], uvIdToVtxId)
        out = tf.get_default_session().run(pnts) + meanPnts

        # pnts = img2pnts(image)
        # out = tf.get_default_session().run(pnts)
        # meanObj.save("output%d.obj"%seed_idx, out)

    t3 = time.time() - t;t=time.time()

    print("tmark", t3)

    meanObj.save("output%d.obj"%seed_idx, out)

# print("elapsed time", time.time() - t)



# cmd = 'ngc batch run --name "sg3d-tf-geom-v3-g%d" --preempt RUNONCE --ace nv-us-west-2 --instance dgx1v.32g.4.norm --commandline "cd /ws/jseo-tf;python train.py --outdir=./geom-train/v3-g%d --gpus=4 --data=data/rasters_tfr --dtype=float32 --gamma=%d --res=256" --result /ws/jseo-tf/geom-train/v3-g%d --image "nvidian/ct/stylegan3d:tf-0.2" --org nvidian --team ct --workspace mdezHHJGSoifZ7iMwGoSEw:/ws:RW'

# for g in (50, 25, 10, 5, 2, 1):
#     print(cmd%(g,g,g,g))



# ngc batch run --name "sg3d-tf-geom-v4-sg2" --preempt RUNONCE --ace nv-us-west-2 --instance dgx1v.32g.8.norm --commandline "cd /ws/jseo-tf;python train.py --outdir=./geom-train/v4-sg2 --gpus=8 --data=data/rasters_tfr --dtype=float32 --res=256 --cfg stylegan2" --result /ws/jseo-tf/geom-train/v4-sg2 --image "nvidian/ct/stylegan3d:tf-0.2" --org nvidian --team ct --workspace mdezHHJGSoifZ7iMwGoSEw:/ws:RW

# ngc batch run --name "sg3d-tf-geom-v4-sg2-noaug" --preempt RUNONCE --ace nv-us-west-2 --instance dgx1v.32g.8.norm --commandline "cd /ws/jseo-tf;python train.py --outdir=./geom-train/v4-sg2-noaug --gpus=8 --data=data/rasters_tfr --dtype=float32 --res=256 --cfg stylegan2 --aug noaug" --result /ws/jseo-tf/geom-train/v4-sg2-noaug --image "nvidian/ct/stylegan3d:tf-0.2" --org nvidian --team ct --workspace mdezHHJGSoifZ7iMwGoSEw:/ws:RW

