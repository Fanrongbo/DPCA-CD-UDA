import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

class retinex_synthesis(nn.Module):
    def __init__(self,channel=3, sigma=5.0,device=None):
        super(retinex_synthesis, self).__init__()
        # self.size=size
        # Gaussian blur
        kernel_size = int(6 * sigma + 1)  # Rule of thumb for kernel size
        if kernel_size % 2 == 0:  # kernel size should be odd.
            kernel_size += 1
        kernel=self.gaussian_kernelBuild(kernel_size,sigma)
        # kernel
        gaussian_kernel = kernel.repeat(channel, 1, 1, 1)
        self.gaussian_kernel = nn.Parameter(gaussian_kernel, requires_grad=False)
        self.kernel_size=kernel_size
    def gaussian_kernelBuild(self,size: int, sigma: float):
        """
        Create a Gaussian kernel.
        size: int, kernel size
        sigma: float, standard deviation for the Gaussian function
        """
        coords = torch.arange(size).float() - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.reshape(1, 1, -1, 1) * g.reshape(1, 1, 1, -1)
    def retinex_decompose(self,img):
        """
            Decompose image into reflectance and illumination components using Retinex.
            img: Tensor of shape (B, C, H, W)
            sigma: float, standard deviation for Gaussian blur.
            """
        # Convert to float for computation
        img = img.float()
        # Log transform
        img_log = torch.log1p(img)
        illumination = F.conv2d(img_log, self.gaussian_kernel, padding=self.kernel_size // 2,
                                groups=3)
        reflectance = img_log - illumination
        return reflectance, illumination
    def adjust_illumination(self,reflectance, illumination):
        """
            Combine reflectance and illumination and then perform inverse log transform.
            reflectance: Tensor of shape (B, C, H, W)
            illumination: Tensor of shape (B, C, H, W)
            """
        img_log = reflectance + illumination
        # print(img_log.shape)
        img_corrected = torch.expm1(img_log)
        img_corrected = torch.clamp(img_corrected, 0, 1)  # Ensure values are between 0 and 1
        return img_corrected
    def outpp(self,input):
        img_corrected = torch.expm1(input)
        img_corrected = torch.clamp(img_corrected, 0, 1)  # Ensure values are between 0 and 1
        return img_corrected
    def forward(self,background,insatance,draw=False):
        reflectance1, illumination1 = self.retinex_decompose(background)
        reflectance2, illumination2 = self.retinex_decompose(insatance)
        img_corrected=self.adjust_illumination(reflectance2,illumination1)
        if draw:
            H1 = self.outpp(reflectance1)
            print('H1',H1.shape)
            L1 = self.outpp(illumination1)
            H2 = self.outpp(reflectance2)
            L2 = self.outpp(illumination2)
            return img_corrected,[H1,L1,H2,L2]

        else:
            return img_corrected

