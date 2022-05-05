import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

'''
Generalized intersection over union loss 
src: https://cs.adelaide.edu.au/~javen/talk/ML17_GIoU-CVPR.pdf
idea page: 33
algorithm page: 35
'''

# Input [batch * max_ob * (x y w h)]
def giou_loss(predicted, target, reduction = "mean"):
    
    area_p = predicted[:,:,2] * predicted[:,:,3]
    area_t = target[:,:,2] * target[:,:,3]
    
    p_x2 = predicted[:,:,0] + predicted[:,:,2]
    t_x2 = target[:,:,0] + target[:,:,2]
    
    p_y2 = predicted[:,:,1] + predicted[:,:,3]
    t_y2 = target[:,:,1] + target[:,:,3]
    
    x1s = torch.stack((predicted[:,:,0], target[:,:,0]), 2)
    y1s = torch.stack((predicted[:,:,1], target[:,:,1]), 2)
    x2s = torch.stack((p_x2, t_x2), 2)
    y2s = torch.stack((p_y2, t_y2), 2)
                      
    
    x1_I = torch.max(x1s, 2)[0]
    y1_I = torch.max(y1s, 2)[0]
    x2_I = torch.min(x2s, 2)[0]
    y2_I = torch.min(y2s, 2)[0]
    
    #print(f'x1_I size {x1_I.size()}')
    #print(f'x2_I size {x2_I.size()}')
    zeros = torch.zeros(x1_I.size()) 
    
    area_I = torch.where((x2_I > x1_I) * (y2_I > y1_I), (x2_I - x1_I) * (y2_I - y1_I), zeros) 
    #print(f'area_I size {area_I.size()}')
    #print(I)
    
    x1_C = torch.min(x1s, 2)[0]
    y1_C = torch.min(y1s, 2)[0]
    x2_C = torch.max(x2s, 2)[0]
    y2_C = torch.max(y2s, 2)[0]
    
    area_C = (x2_C - x1_C) * (y2_C - y1_C)
    #print(area_C)
    
    U = area_p + area_t - area_I
    
    IoU = area_I / U
    
    GIoU = IoU - (area_C - U)/(area_C)
    
    LossGIoU = 1 - GIoU
    
    if reduction == 'mean':
        return torch.mean(LossGIoU)
    if reduction == 'sum':
        return torch.sum(LossGIoU)
    if reduction == 'none':
        return LossGIoU
    else:
        print(f'Unknown reduction in giou_loss: {reduction} (available: mean, sum, none)')
        return None


if __name__ == '__main__':
    
    batch = 1
    objs = 1
    predic = torch.rand(batch, objs, 4)
    target = torch.rand(batch, objs, 4)
    #target = predic
    
    print(f'predicted size {predic.size()}')
    print(f'target size {target.size()}')
    
    loss = giou_loss(predic, target, reduction = 'none')
    print(loss)
    
    # TEST PLOT
    plt.xlim(0,2)
    plt.ylim(0,2)
    ax = plt.gca()
    
    for b in range(batch):
        for o in range(objs):
            r = Rectangle((predic[b,o,0],predic[b,o,1]),predic[b,o,2], predic[b,o,3], ec = 'green', fc="none")
            ax.add_patch(r)
            
            r = Rectangle((target[b,o,0],target[b,o,1]),target[b,o,2], target[b,o,3], ec = 'red', fc="none")
            ax.add_patch(r)
            
            plt.text((predic[b,o,0] + predic[b,o,2]/2 + target[b,o,0] + target[b,o,2]/2)/2, (predic[b,o,1] + target[b,o,1] + predic[b,o,3]/2 + target[b,o,3]/2)/2, '{:.3f}'.format(loss[b,o]), fontsize=12)
            
    plt.show()
            
            
        
    
    
