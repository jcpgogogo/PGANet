
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# define the high-order finite difference kernels
lapl_op = [[[[    0,   0, -1/12,   0,     0],
             [    0,   0,   4/3,   0,     0],
             [-1/12, 4/3,    -5, 4/3, -1/12],
             [    0,   0,   4/3,   0,     0],
             [    0,   0, -1/12,   0,     0]]]]

partial_y = [[[[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [1/12, -8/12, 0, 8/12, -1/12],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]]]]

partial_x = [[[[0, 0, 1/12, 0, 0],
               [0, 0, -8/12, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 8/12, 0, 0],
               [0, 0, -1/12, 0, 0]]]]
class Conv2dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=3, name=''):
        super(Conv2dDerivative, self).__init__()

        self.resol = resol  # constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size, 
            1, padding=0, bias=False)

        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.FloatTensor(DerFilter), requires_grad=False)  

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol


class Conv1dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=3, name=''):
        super(Conv1dDerivative, self).__init__()

        self.resol = resol  # $\delta$*constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv1d(self.input_channels, self.output_channels, self.kernel_size, 
            1, padding=0, bias=False)
        
        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.FloatTensor(DerFilter), requires_grad=False)  

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol


class loss_generator(nn.Module):
    ''' Loss generator for physics loss '''

    def __init__(self, dt = (10.0/200), dx = (20.0/128)):
        ''' Construct the derivatives, X = Width, Y = Height '''
       
        super(loss_generator, self).__init__()

        # spatial derivative operator
        self.laplace = Conv2dDerivative(
            DerFilter = lapl_op,
            resol = (dx**2),
            kernel_size = 5,
            name = 'laplace_operator').cuda()

        self.dx = Conv2dDerivative(
            DerFilter = partial_x,
            resol = (dx*1),
            kernel_size = 5,
            name = 'dx_operator').cuda()

        self.dy = Conv2dDerivative(
            DerFilter = partial_y,
            resol = (dx*1),
            kernel_size = 5,
            name = 'dy_operator').cuda()

        # temporal derivative operator
        self.dt = Conv1dDerivative(
            DerFilter = [[[-1,0,  1]]],
            resol = (dt*2),
            kernel_size = 3,
            name = 'partial_t').cuda()

    def get_phy_Loss(self, output):

        # spatial derivatives
        u0=output[1:-1, 0:1, :, :]
        laplace_u = self.laplace(u0)  # [t,c,h,w]
        #laplace_v = self.laplace(output[1:-1, 1:2, :, :])

        u_x = self.dx(output[1:-1, 0:1, :, :])
        #u_xx=self.dx(u_x[1:, 0:1, :, :])
        u_y = self.dy(output[1:, 0:1, :, :])
        #u_yy=self.dy(u_y[1:, 0:1, :, :])
        #v_x = self.dx(output[1:-1, 1:2, :, :])
        #v_y = self.dy(output[1:-1, 1:2, :, :])
        uabs=torch.abs(laplace_u)
        uexp=torch.exp(-torch.pow(uabs/5, 2))
        fxy=torch.mul(uexp,laplace_u)
        fxy2= self.laplace(fxy[:, 0:1, :, :]) 
        # fxy=torch.sqrt(torch.pow(u_xx, 2)+torch.pow(u_yy, 2))
        # fxy2=torch.mul(fxy,laplace_u[:, 0:1, 2:-2, 2:-2])

        # temporal derivative - u
        u = output[:, 0:1, 2:-2, 2:-2]
        lent = u.shape[0]
        lenx = u.shape[3]
        leny = u.shape[2]
        u_conv1d = u.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        u_conv1d = u_conv1d.reshape(lenx*leny,1,lent)
        u_t = self.dt(u_conv1d)  # lent-2 due to no-padding
        u_t = u_t.reshape(leny, lenx, 1, lent-2)
        u_t = u_t.permute(3, 2, 0, 1)  # [step-2, c, height(Y), width(X)]

        # temporal derivative - v
        # v = output[:, 1:2, 2:-2, 2:-2]
        # v_conv1d = v.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        # v_conv1d = v_conv1d.reshape(lenx*leny,1,lent)
        # v_t = self.dt(v_conv1d)  # lent-2 due to no-padding
        # v_t = v_t.reshape(leny, lenx, 1, lent-2)
        # v_t = v_t.permute(3, 2, 0, 1)  # [step-2, c, height(Y), width(X)]

        u = output[1:-1, 0:1, 2:-2, 2:-2]  # [t, c, height(Y), width(X)]
        #v = output[1:-1, 1:2, 2:-2, 2:-2]  # [t, c, height(Y), width(X)]

        assert laplace_u.shape == u_t.shape
        #assert u_t.shape == v_t.shape
        assert laplace_u.shape == u.shape
        #assert laplace_v.shape == v.shape

        R = 200.0

        # 2D burgers eqn
        #fxu=torch.sqrt(laplace_u)
        f_u = u_t[:, 0:1, 2:-2, 2:-2] - fxy2
        #f_v = v_t + u * v_x + v * v_y - (1/R) * laplace_v

        return f_u#, f_v


def compute_loss(output, loss_func):
    ''' calculate the phycis loss '''
    
    # Padding x axis due to periodic boundary condition
    # shape: [t, c, h, w]
    output = torch.cat((output[:, :, :, -2:], output, output[:, :, :, 0:3]), dim=3)

    # Padding y axis due to periodic boundary condition
    # shape: [t, c, h, w]
    output = torch.cat((output[:, :, -2:, :], output, output[:, :, 0:3, :]), dim=2)

    # get physics loss
    mse_loss = nn.MSELoss()
    f_u = loss_func.get_phy_Loss(output)
    loss =  mse_loss(f_u, torch.zeros_like(f_u).cuda()) 

    return loss

def pdelossfun(noisy,denoisy):
    dt = 0.5
    dx = 1.0
    loss_func = loss_generator(dt, dx)
    inputnoise=noisy#.cpu()
    outreshape=denoisy#.cpu()
    # inputnoise=torch.squeeze(noisy,0).cpu()
    # outreshape=torch.squeeze(denoisy,0).cpu()
    output1= np.stack((inputnoise[0:1,:,:], inputnoise[0:1,:,:],outreshape.detach()[0:1,:,:],outreshape.detach()[0:1,:,:])) 
    # output2= np.stack((inputnoise[1:2,:,:], inputnoise[1:2,:,:],outreshape.detach()[1:2,:,:],outreshape.detach()[1:2,:,:])) 
    # output3= np.stack((inputnoise[2:3,:,:], inputnoise[2:3,:,:],outreshape.detach()[2:3,:,:],outreshape.detach()[2:3,:,:]))
    #output1= np.stack(( inputnoise[0:1,:,:],outreshape.detach()[0:1,:,:],outreshape.detach()[0:1,:,:])) 
    #output2= np.stack(( inputnoise[1:2,:,:],outreshape.detach()[1:2,:,:],outreshape.detach()[1:2,:,:])) 
    #output3= np.stack(( inputnoise[2:3,:,:],outreshape.detach()[2:3,:,:],outreshape.detach()[2:3,:,:])) 
       
    losspde1 = compute_loss(torch.tensor(output1).cuda().float(), loss_func) 
    # losspde2 = compute_loss(torch.tensor(output2).cuda().float(), loss_func)
    # losspde3 = compute_loss(torch.tensor(output3).cuda().float(), loss_func) 
    # return (losspde1+losspde2+losspde3)*0.005 
    return (losspde1)*0.005 




class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class up(nn.Module):
    def __init__(self, in_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = x2 + x1
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.fcn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.fcn(x)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.inc = nn.Sequential(
            single_conv(6, 64),
            single_conv(64, 64)
        )

        self.down1 = nn.AvgPool2d(2)
        self.conv1 = nn.Sequential(
            single_conv(64, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )

        self.down2 = nn.AvgPool2d(2)
        self.conv2 = nn.Sequential(
            single_conv(128, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256)
        )

        self.up1 = up(256)
        self.conv3 = nn.Sequential(
            single_conv(128, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )

        self.up2 = up(128)
        self.conv4 = nn.Sequential(
            single_conv(64, 64),
            single_conv(64, 64)
        )

        self.outc = outconv(64, 3)

    def forward(self, x):
        inx = self.inc(x)

        down1 = self.down1(inx)
        conv1 = self.conv1(down1)

        down2 = self.down2(conv1)
        conv2 = self.conv2(down2)

        up1 = self.up1(conv2, conv1)
        conv3 = self.conv3(up1)

        up2 = self.up2(conv3, inx)
        conv4 = self.conv4(up2)

        out = self.outc(conv4)
        return out


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fcn = FCN()
        self.unet = UNet()
    
    def forward(self, x):
        noise_level = self.fcn(x)
        concat_img = torch.cat([x, noise_level], dim=1)
        out = self.unet(concat_img) + x
        return noise_level, out


class fixed_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, out_image, gt_image, est_noise, gt_noise, if_asym):
        l2_loss = F.mse_loss(out_image, gt_image)

        asym_loss = torch.mean(if_asym * torch.abs(0.3 - torch.lt(gt_noise, est_noise).float()) * torch.pow(est_noise - gt_noise, 2))

        h_x = est_noise.size()[2]
        w_x = est_noise.size()[3]
        count_h = self._tensor_size(est_noise[:, :, 1:, :])
        count_w = self._tensor_size(est_noise[:, :, : ,1:])
        h_tv = torch.pow((est_noise[:, :, 1:, :] - est_noise[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((est_noise[:, :, :, 1:] - est_noise[:, :, :, :w_x-1]), 2).sum()
        tvloss = h_tv / count_h + w_tv / count_w

        dt = 0.5
        dx = 1.0
        loss_func = loss_generator(dt, dx) 
        inputnoise=torch.squeeze(gt_image+gt_noise,0).cpu()
        outreshape=torch.squeeze(out_image,0).cpu()
        output1= np.stack((inputnoise[0:1,:,:], inputnoise[0:1,:,:],outreshape.detach()[0:1,:,:],outreshape.detach()[0:1,:,:])) 
        output2= np.stack((inputnoise[1:2,:,:], inputnoise[1:2,:,:],outreshape.detach()[1:2,:,:],outreshape.detach()[1:2,:,:])) 
        output3= np.stack((inputnoise[2:3,:,:], inputnoise[2:3,:,:],outreshape.detach()[2:3,:,:],outreshape.detach()[2:3,:,:]))
        #output1= np.stack(( inputnoise[0:1,:,:],outreshape.detach()[0:1,:,:],outreshape.detach()[0:1,:,:])) 
        #output2= np.stack(( inputnoise[1:2,:,:],outreshape.detach()[1:2,:,:],outreshape.detach()[1:2,:,:])) 
        #output3= np.stack(( inputnoise[2:3,:,:],outreshape.detach()[2:3,:,:],outreshape.detach()[2:3,:,:])) 
       
        losspde1 = compute_loss(torch.tensor(output1).cuda().float(), loss_func) 
        losspde2 = compute_loss(torch.tensor(output2).cuda().float(), loss_func)
        losspde3 = compute_loss(torch.tensor(output3).cuda().float(), loss_func)     
       # total_loss = F.mse_loss(out, img_noisy_torch)#+(losspde1+losspde2+losspde3)*0.01

        loss = l2_loss +  0.5 * asym_loss + 0.05 * tvloss  +(losspde1+losspde2+losspde3)*0.007

        return loss

    def _tensor_size(self, t):
        return t.size()[1]*t.size()[2]*t.size()[3]