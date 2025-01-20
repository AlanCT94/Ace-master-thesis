import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from PIL import Image
from dataset import CamLocDataset
from torch.cuda.amp import autocast


def ifgsm(model, X, y_true, epsilon, iterations,image_og,dataset,counter,L1_Loss,mean=0.4,sd=0.25,alpha=1):
    """ 
    
    Create adversarial attacks with a define number of iterations
    model:      The ML model to make a prediction. 'Network' in ACE test_ace
    X:          The image tensor from the testset_loader image_B1HW
    y_true:     The true target. 3D coordinates in this case gt_sc. 
    epsilon:    The parameter from IFGSM. a list
    iterations: Number of iterations to produce the adversarial attacks. a int
    image_og:   The original .png image loaded from the function _load_image
    dataset =   The list to add new rows. To be converted at the end to a pandas dataframe.
    counter =   Is the number of the image to know which sequence is the image from
    alpha=1     Tha parameter from IFGSM used on the paper by Kurakin et al.
                Alpha equal to 1 to change the value of the pixel by 1 on each 
                step.

    """
    X_adv = X # this is the image_B1HW tensor from testset_loader
    grad_list = []
    
    for eps in epsilon:
        eps1 = round(eps,1)
        img_min_eps = X - eps1
        img_max_eps = X + eps1
        
        for step in range(iterations):
            print(f'counter:{counter},epsilon:{step}')
            # 1. Copmute the gradient with respect the input image ##################
            with autocast(enabled=True):
                scene_coordinates_B3HW = model(X_adv) # Predict the coord. with the X_adv
            
            Loss_fn = torch.nn.L1Loss() if L1_Loss else torch.nn.MSELoss()
            Ln_loss = Loss_fn(y_true, scene_coordinates_B3HW)

            # Backpropagation
            model.zero_grad()
            Ln_loss.backward()

            # Gradient
            gradient = X_adv.grad.clone()
            grad_list.append(gradient)
            
            # 2. Compute the adversarial perturbation #########################
            
            perturbation = alpha*torch.sign(gradient)

            X_adv = X_adv + perturbation # X' in the paper
            
            X_adv = torch.clamp(X_adv, min=img_min_eps, max=img_max_eps).clone().detach().requires_grad_(True)

            #############################################################################

            X_adv_to_sv = X_adv.clone() - X # Only the perturbation substract the original image
         
            X_adv_to_sv = X_adv_to_sv.clone()

    
            img_to_sv = torch.cat([X_adv_to_sv,X_adv_to_sv,X_adv_to_sv],dim=0).squeeze().detach()
            img_to_sv = torch.reshape(img_to_sv,(480,640,3))
            X_adv_numpy = img_to_sv.cpu().numpy() # Change to numpy array

            X_adv_numpy = (X_adv_numpy*sd)*255 #+mean 
            # Save the adversarial example
            image = np.array(image_og)

            adversarial_RGB = image + X_adv_numpy
            
            adversarial_RGB = np.clip(adversarial_RGB,0,255)

            # Save the adversarial example
            a = 5 if counter >= 10 else 3
            
            img_to_png = Image.fromarray(adversarial_RGB.astype(np.uint8))
            save_path_adv_ex = f"/home/alan/Desktop/ace_mt_Copy/datasets/7scenes_chess/L{1 if L1_Loss else 2}_IFGSM/rgb/seq-0{a}-frame-{counter:06}-eps-{eps1}-iteration-{step}.color.png"
            img_to_png.save(save_path_adv_ex)
            
            dataset.append({'sequence': a, 'frame': counter, 'epsilon': eps1, 'iteration': step+1 })
    torch.save(grad_list,"gradients_L2IFGSM.pt") # L1 or L2        
    return dataset