""" Full assembly of the parts to form the complete network """
import math
import random
import torch.nn as nn
import torch
import numpy as np
import time
# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import unet_parts as part


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = part.DoubleConv(in_channels, 64)
        self.down1 = part.Down(64, 128)
        self.down2 = part.Down(128, 256)
        self.down3 = part.Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = part.Down(512, 1024 // factor)
        self.up1 = part.Up(1024, 512 // factor, bilinear)
        self.up2 = part.Up(512, 256 // factor, bilinear)
        self.up3 = part.Up(256, 128 // factor, bilinear)
        self.up4 = part.Up(128, 64, bilinear)
        self.outc = part.OutConv(64, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
    def fit(self, train_input_batch,train_output_batch,test_input_batch,test_output_batch, EPOCHS,opt,gamma,sheduler_period,batch_size=16):
        startTime = time.time()
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
        train_loss_list = []
        test_loss_list = []
#        MAE_test_loss_list = []
        
        #Training loop
        for e in range(0, EPOCHS):
            # loop over the training set
            n = 0
            nmax=train_input_batch.shape[0]/batch_size
            
            perm_train = torch.randperm(train_output_batch.shape[0])
            perm_test = torch.randperm(test_output_batch.shape[0])
            
            train_input_batch = train_input_batch[perm_train]
            train_output_batch = train_output_batch[perm_train]
            test_input_batch = test_input_batch[perm_test]
            test_output_batch = test_output_batch[perm_test]
            
#           
            
            if e%sheduler_period==0:
                scheduler.step()
            
            #bath size split
            for (x,y) in zip(train_input_batch.split(batch_size),train_output_batch.split(batch_size)):

                #training
                
                n=n+1
                
                
#                print("pred.shape = ",pred.shape)
#                print("y.shape = ",y.shape)
                
                #ùetrics
               
#                MAE_train = torch.abs((y.cuda()-pred)).mean()
#                #
#                ssim_train = -self.ssim(pred,y.cuda())
#                
#                mge_train = self.MGE_loss(pred,y.cuda())
#                
#                lap = self.Laplacian(pred,y)
#                
#                loss = 0.1*mge_train + 0.1*ssim_train + 0.1*lap + MAE_train + 0.1
                
                pred = self.forward(x.to(device))
                loss =((y.to(device)-pred)**2).mean()
               
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_loss_list.append(loss.item())
                
                
                
                # zero out the gradients, perform the backpropagation step,
                # and update the weights
                
                
#                
#                randint=random.randrange(0,len(test_output_batch.split(batch_size)))    #rend test sample permutation
#                with torch.no_grad():
#                    pred_test=self.forward(test_input_batch.split(batch_size)[randint].cuda())
#                MSE_test=((test_output_batch.split(batch_size)[randint].cuda()-pred_test)**2).mean()
#                train_loss_list.append(MSE_train.item())
#                ssim_test=-ssim(self.forward(test_input_batch.split(batch_size)[randint].cuda()),test_output_batch.split(batch_size)[randint].cuda())
#
#                mge_test=MGE_loss(self.forward(test_input_batch.split(batch_size)[randint].cuda()),test_output_batch.split(batch_size)[randint].cuda(),batch_size=batch_size)
#                #test_loss_list.append(test_loss.item())
#                test_loss=0.1*mge_test+0.1*ssim_test+MSE_test+0.1
#                test_loss_list.append(MSE_test.item())
                
                
                
#                if loss.item()<test_loss.item():
#                    overfitting_counter+=1
#                else:
#                    overfitting_counter=0
#                if overfitting_counter>30:
#                    print("overfitting_breaked")
#                    break
                
                
             
                
                ##test loss##
                randint=random.randrange(0,len(test_output_batch.split(batch_size)))  # select random indice of batch
                with torch.no_grad():
                    pred_test = self.forward(test_input_batch.split(batch_size)[randint].to(device))
                test_target = test_output_batch.split(batch_size)[randint].to(device)
#                MAE_test = torch.abs((test_target-pred_test)).mean()
#                ssim_test = -self.ssim(pred_test,test_target)
#                mge_test = self.MGE_loss(pred_test,test_target)
#                lap_test = self.Laplacian(pred_test,test_target)
                test_loss=((pred_test-test_target)**2).mean()
#                test_loss = 0.1*mge_test + 0.1 * ssim_test + 0.1*lap_test + MAE_test + 0.1
                test_loss_list.append(test_loss.item())
#                MAE_test_loss_list.append(MAE_test.item())
                
                endTime = time.time()
                Time=endTime-startTime
                print("\nTraining : time elapsed = ",Time,"/",(Time/(e*nmax+n))*EPOCHS*nmax,"\nEpoch = ",e,"/",EPOCHS, " nbatch = ",n,"/",nmax)
                print("Loss train =",loss.item())
#                print("MGE train= ",mge_train.item())
#                print("MSE train= ",MAE_train.item())
#                print("loss test  =",test_loss.item())
#                print("MGE test= ",mge_test.item())
#                print("MSE test= ",MAE_test.item())
               
               
        return train_loss_list,test_loss_list
    
    def predict(self,input_batch,output_batch,rescale):
        
        pred_list = []
        visu_list=[]
        n = 0
        nmax = input_batch.shape[0]
        for im_input,im_output in zip(input_batch,output_batch):
            n = n+1
            print(" predict n = ",n,"/",nmax)
            im_input = im_input.unsqueeze(0)
            pred = self.forward(im_input.to(device))[0]
            if torch.cuda.is_available():
                pred=pred.cpu()
          
            #im_output_visu = torch.cat((nn.functional.interpolate(im_input,size=(214,267),mode='bicubic')[0],pred,im_output),dim=2)
            pred = (pred.permute(1,2,0).detach().numpy()*rescale)[:,:,0]
            interp=(nn.functional.interpolate(im_input,size=(214,267),mode='bicubic')[0].permute(1,2,0).detach().numpy()*rescale)[:,:,0]
            im_input=(im_input[0].permute(1,2,0).detach().numpy()*rescale)[:,:,0]            
            
            pred_list.append(pred.astype(np.uint8))
            im_output=(im_output.permute(1,2,0).detach().numpy()*rescale)[:,:,0]
            visu_list.append([im_input,interp,pred,im_output])
            #saving image
            #im_out.save(f"%s/%s-predicted.jpeg" %(dir_name,name))
        return np.asarray(pred_list),np.asarray(visu_list)
    def gaussian(self,window_size, sigma):
        """
        Generates a list of Tensor values drawn from a gaussian distribution with standard
        diviation = sigma and sum of all elements = 1.
    
        Length of list = window_size
        """    
        gauss =  torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
        
    def create_window(self,window_size, channel=3):
    
        # Generate an 1D tensor containing values sampled from a gaussian distribution
        _1d_window = self.gaussian(window_size=window_size, sigma=1.5).unsqueeze(1)
        
        # Converting to 2D  
        _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
         
        window = torch.Tensor(_2d_window.expand(channel, 1, window_size, window_size).contiguous())
    
        return window
    
                
    def ssim(self,img1, img2, window_size=11, val_range=255, window=None, size_average=True, full=False):
    
        L = val_range # L is the dynamic range of the pixel values (255 for 8-bit grayscale images),
    
        pad = window_size // 2
        
        try:
            _, channels, height, width = img1.size()
        except:
            channels, height, width = img1.size()
    
        # if window is not provided, init one
        if window is None: 
            real_size = min(window_size, height, width) # window should be atleast 11x11 
            window = self.create_window(real_size, channel=channels).to(img1.device)
        
        # calculating the mu parameter (locally) for both images using a gaussian filter 
        # calculates the luminosity params
        mu1 = nn.functional.conv2d(img1, window, padding=pad, groups=channels)
        mu2 = nn.functional.conv2d(img2, window, padding=pad, groups=channels)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2 
        mu12 = mu1 * mu2
    
        # now we calculate the sigma square parameter
        # Sigma deals with the contrast component 
        sigma1_sq = nn.functional.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
        sigma2_sq = nn.functional.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
        sigma12 =  nn.functional.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu12
    
        # Some constants for stability 
        C1 = (0.01*L ) ** 2  # NOTE: Removed L from here (ref PT implementation)
        C2 = (0.03*L ) ** 2 
    
        contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        contrast_metric = torch.mean(contrast_metric)
    
        numerator1 = 2 * mu12 + C1  
        numerator2 = 2 * sigma12 + C2
        denominator1 = mu1_sq + mu2_sq + C1 
        denominator2 = sigma1_sq + sigma2_sq + C2
    
        ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)
    
        if size_average:
            ret = ssim_score.mean() 
        else: 
            ret = ssim_score.mean(1).mean(1).mean(1)
        
        if full:
            return ret, contrast_metric
        
        return ret      
    
    def MGE_loss(self,pred,target):
        output_channels=self.output_channels
        x_filter = torch.Tensor([[[[-1, 0, 1], [-2, 1, 2], [-1, 0, 1]]]]).repeat(1,output_channels,1,1).to(device)
        y_filter = torch.Tensor([[[[-1,-2,-1], [0,0,0], [1,2,1]]]]).repeat(1,output_channels,1,1).to(device)
        pad = nn.ReplicationPad2d(1)
        pred = pad(pred)
        target = pad(target)
        
        if torch.cuda.is_available():
            pred_gx = nn.functional.conv2d(pred,x_filter,stride=1, padding=0).cpu().detach().numpy()
            pred_gy = nn.functional.conv2d(pred, y_filter,stride=1, padding=0).cpu().detach().numpy()
            target_gx = nn.functional.conv2d(target, x_filter, padding=0).cpu().detach().numpy()
            target_gy = nn.functional.conv2d(target, y_filter, padding=0).cpu().detach().numpy()
        else : 
            pred_gx = nn.functional.conv2d(pred,x_filter,stride=1, padding=0).detach().numpy()
            pred_gy = nn.functional.conv2d(pred, y_filter,stride=1, padding=0).detach().numpy()
            target_gx = nn.functional.conv2d(target, x_filter, padding=0).numpy()
            target_gy = nn.functional.conv2d(target, y_filter, padding=0).numpy()
        g_pred = np.sqrt(pred_gx**2 + pred_gy**2)
        g_target = np.sqrt(target_gx**2 + target_gy**2)    
        return ((g_pred - g_target)**2).mean()

    def Laplacian(self,pred,target):
        output_channels = self.output_channels
        filters = torch.tensor([[[[0,1,0],[1,-4,1],[0,1,0]]]]).repeat(1,output_channels,1,1).float().to(device)
        pad = nn.ReplicationPad2d(1)
        pred = pad(pred)
        target = pad(target)
        if torch.cuda.is_available():
            lpred = nn.functional.conv2d(pred,filters,stride=1,padding=0).cpu().detach().numpy()
            ltarget = nn.functional.conv2d(pred,filters,stride=1,padding=0).cpu().detach().numpy()
        else :
            lpred = nn.functional.conv2d(pred,filters,stride=1,padding=0).detach().numpy()
            ltarget = nn.functional.conv2d(pred,filters,stride=1,padding=0).detach().numpy()
        return np.abs(lpred - ltarget).mean()
