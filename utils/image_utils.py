import matplotlib.pyplot as plt
import numpy as np
import torch
from dataloaders.utils import img_denormalize
from dataloaders.utils import decode_segmap
from utils.get_model import get_model
from utils.get_dataset import get_val_dataset


"""
Python implementation of the color map function for the PASCAL VOC data set. 
Official Matlab version can be found in the PASCAL VOC devkit 
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
"""

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


def transform_to_image(inp, mean_image, std_image):

    """ Undoes changes in the mean and standard deviation """

    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array(mean_image)
    std = np.array(std_image)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    return inp


def visualize_images(**images):
    """PLot images in one row."""
    plt.close()
    n = len(images)
    #plt.figure(figsize=(12, 3))
    #
    # for i, (name, image) in enumerate(images.items()):
    #     plt.subplot(1, n, i + 1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.title(' '.join(name.split('_')).title())
    #     plt.imshow(image)
    #
    # plt.show(block=False)
    # plt.pause(1 / 20)

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.colors import NoNorm

    fig, axs = plt.subplots(1, n, figsize=(15, 4))
   # cm = ['viridis', 'viridis', 'viridis', 'viridis', 'viridis', 'viridis','bone', 'bone', 'bone']
   # cm = ['viridis', 'viridis', 'viridis', 'viridis', 'viridis', 'viridis', 'viridis', 'viridis', 'viridis']

    out = [(k, v) for k, v in images.items()]
    for col in range(n):
            ax = axs[col]
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())

            if col <= 5:
                pcm = ax.imshow(out[col][1], cmap='viridis')

            if col > 5:

                #pcm = ax.imshow(out[col][1], cmap='bone', norm=NoNorm())
                pcm = ax.imshow(out[col][1], cmap='viridis')
                axins = inset_axes(ax,
                                    width="10%",  # width = 50% of parent_bbox width
                                    height="300%",  # height : 5%
                                    loc='right')

                fig.colorbar(pcm, cax=axins, orientation="vertical")
                #axins.xaxis.set_ticks_position("bottom")

                #divider = make_axes_locatable(ax)
                #cax = divider.append_axes('right', size='10%', pad=0.03)
                #fig.colorbar(pcm, cax=cax, orientation='vertical')
                #fig.colorbar(pcm,  orientation='vertical')

    #fig.tight_layout(pad=0.05)
    plt.show(block=False)
    plt.pause(1 / 20)




def process_visualize_image(dataset, device, model):

    n = np.random.choice(len(dataset))

    sample = dataset[n]
    image = sample['image']
    gt_mask = sample['label']
    #edges = sample['edges']
    print(sample['name'])

    x_tensor = image.to(device).unsqueeze(0)

    pr = model(x_tensor)

    if isinstance(pr, list):
        pr = pr[0]

    if type(pr) is tuple:
        #prob = pr[4]
        pr = pr[0]

    _, pr_mask = torch.max(pr, dim=1)
    pr_mask = pr_mask.squeeze()

    image_original = img_denormalize(image.cpu().numpy(), dataset.mean, dataset.std)
    object_channel = pr[0, 1, :, :]
    object_channel = object_channel.detach()

    object_channel_2 = pr[0, 2, :, :].detach()
    #object_channel_2 = pr[0, 0, :, :].detach()


    #epsilon = 0.000000000000000000000000000001
    epsilon = 1.0e-32
    nll = pr.detach()
    true_prob = torch.exp(nll) + epsilon
    # bottom = true_prob[0, 0, :, :]*prob[0] + true_prob[0, 1, :,:]*prob[1] + true_prob[0,2,:,:]*prob[2]
    # class0 = true_prob[0, 0, :, :] * prob[0] / bottom
    # class1 = true_prob[0, 1, :, :] * prob[1] / bottom
    # class2 = true_prob[0, 2, :, :] * prob[2] / bottom

    bottom = true_prob[0, 0, :, :] + true_prob[0, 1, :, :] + true_prob[0, 2, :, :]
    class0 = true_prob[0, 0, :, :] / bottom
    class1 = true_prob[0, 1, :, :] / bottom
    class2 = true_prob[0, 2, :, :] / bottom
    # print(torch.max(class0), torch.min(class0))
    # print(torch.max(class1), torch.min(class1))
    # print(torch.max(class2), torch.min(class2))

    return image_original, pr_mask, gt_mask, object_channel, object_channel_2, (class0, class1, class2)


def visualize_results(train_dataset, valid_dataset, model, device):
    model.eval()

    if valid_dataset is None:
        valid_dataset = train_dataset

    image_original, pr_mask, gt_mask, pr, pr2, cl1 = process_visualize_image(train_dataset, device, model)

    gt_out = np.array(gt_mask).astype(np.uint8)

    dataset_name = train_dataset.name()
    segmap_gt = decode_segmap(gt_out, dataset=dataset_name)

    #V,H = compute_edge_mask(torch.unsqueeze(image_original,0), sigma=0.1)

    if valid_dataset is not None:
        image_original_val, pr_mask_val, gt_mask_val, pr, pr2, cl = process_visualize_image(valid_dataset, device, model)
        gt_val_out = np.array(gt_mask_val).astype(np.uint8)
        segmap_gt_val = decode_segmap(gt_val_out, dataset=dataset_name)

        visualize_images(
            image_train = image_original,
            gt_train=segmap_gt,
            predict_train=decode_segmap(pr_mask.cpu().numpy(), dataset_name),
            image_valid=image_original_val,
            gt_val=segmap_gt_val,
            predict_valid = decode_segmap(pr_mask_val.cpu().numpy(), dataset_name),
#            pr=pr.cpu().numpy(),
 #           pr2=pr2.cpu().numpy(),
            cl0=cl[0].cpu().numpy(),
            cl1=cl[1].cpu().numpy(),
            cl2=cl[2].cpu().numpy()
     #       edgeV=V.squeeze().cpu().numpy(),
      #      edgeH=H.squeeze().cpu().numpy()
        )
