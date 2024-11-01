import torch
import torchvision

# from torchvision.transforms import v2 as transforms

from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate


class Trans:
    tsr = transforms.ToTensor()
    pil = transforms.ToPILImage()
    scale_rgb = transforms.Lambda(lambda image: image / 255.0)
    permute = transforms.Lambda(lambda tsr: tsr.permute(2, 0, 1))
    
    cifar10_norm = transforms.Normalize(mean=[0.4913996756076813, 0.48215848207473755, 0.44653090834617615], std=[0.24703224003314972, 0.24348513782024384, 0.26158785820007324], inplace=True)
    cifar100_norm = transforms.Normalize(mean=[0.5070751905441284, 0.48654890060424805, 0.44091784954071045], std=[0.2673342823982239, 0.2564384639263153, 0.2761504650115967], inplace=True)

    in_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True)
    as_norm = transforms.Normalize(
        mean=[0.18503567576408386, 0.27679356932640076, 0.43360984325408936],
        std=[0.08373230695724487, 0.07494986057281494, 0.06476051360368729],
        inplace=True,
    )
    stl_norm = transforms.Normalize(mean=[0.44671064615249634, 0.4398098886013031, 0.4066464304924011], std=[0.26034098863601685, 0.2565772831439972, 0.2712673842906952], inplace=True)

    np_trance = transforms.Lambda(lambda x: -x)
    color = transforms.Lambda(lambda image: image.convert("RGB"))
    mono = transforms.Grayscale(num_output_channels=1)
    rotate90 = transforms.Lambda(lambda image: rotate(image, 90))
    rotate180 = transforms.Lambda(lambda image: rotate(image, 180))
    rotate270 = transforms.Lambda(lambda image: rotate(image, 270))
    hflip = transforms.RandomHorizontalFlip(p=1)
    vflip = transforms.RandomVerticalFlip(p=1)

    def norm(mean, std):
        transforms.Normalize(mean=mean, std=std, inplace=True)

    def cf_raug(**kwargs):
        torchvision.transforms.RandAugment(**kwargs)

    def repeat_data(n):
        # torch.tileだと、次元数をかえたときにタプルの記述を変えなければいけないため(1d => (n, 1), 2d => (n, 1, 1)、その必要が無いようにcatで実装
        return transforms.Lambda(lambda tensor: torch.cat([tensor for _ in range(n)], dim=0))

    def rotate(th):
        return transforms.Lambda(lambda image: rotate(image, th))

    def resize(h, w):
        return torchvision.transforms.Resize((h, w), antialias=True)  # antialias を True にしないと、warning がでる

    # flipや回転など、PILの状態で操作しなければいけないものもあり、それらはテンソルにする前に行う必要がある。
    rflip = [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    rotate_flip = [transforms.RandomRotation(degrees=(0, 360), interpolation=InterpolationMode.BICUBIC), transforms.RandomHorizontalFlip(p=0.5)]
    light_aug = [transforms.RandomRotation(degrees=(-45, 45), interpolation=InterpolationMode.BICUBIC), transforms.RandomHorizontalFlip(p=0.5)]

    tsr_l = [tsr]

    cifar10_gen = [tsr, cifar10_norm]
    cifar10_crop = [transforms.RandomCrop(32, padding=4, padding_mode="reflect"), transforms.RandomHorizontalFlip(), tsr, cifar10_norm]
    cifar10_git = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), tsr, cifar10_norm]

    cifar100_gen = [tsr, cifar10_norm]
    cifar100_crop = [transforms.RandomCrop(32, padding=4, padding_mode="reflect"), transforms.RandomHorizontalFlip(), tsr, cifar10_norm]
    cifar100_git = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), tsr, cifar10_norm]


    stl_gen_32 = [resize(32, 32), tsr, stl_norm]
    stl_git_32 = [resize(32, 32), transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), tsr, stl_norm]
    stl_gen_64 = [resize(64, 64), tsr, stl_norm]
    stl_git_64 = [resize(64, 64), transforms.RandomCrop(64, padding=4), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), tsr, stl_norm]
    

    in_gen = [torchvision.transforms.Resize((224, 224)), tsr, in_norm]

    as_gen = [tsr, scale_rgb, as_norm]
    as_da = [tsr, scale_rgb, transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), as_norm]
    as_da2 = [tsr, scale_rgb, transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.85, 1.15), interpolation=InterpolationMode.BILINEAR), as_norm]
    as_da3 = [tsr, scale_rgb, transforms.ColorJitter(brightness=0.15, contrast=0.15), transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.85, 1.15), interpolation=InterpolationMode.BILINEAR), as_norm]
    as_da4 = [tsr, scale_rgb, *rflip, transforms.ColorJitter(brightness=0.08, contrast=0.08), transforms.RandomAffine(degrees=15, translate=(0.10, 0.10), scale=(0.93, 1.07), interpolation=InterpolationMode.BILINEAR), as_norm]
