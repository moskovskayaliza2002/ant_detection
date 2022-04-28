def glou_loss(predicted, true): #[b * max_ob * 4]
    p_x1 = predicted[:, 0]
    p_x2 = predicted[:, 1]
    p_y1 = predicted[:, 2]
    p_y2 = predicted[:, 3]
    
    t_x1 = true[:, 0]
    t_x2 = true[:, 1]
    t_y1 = true[:, 2]
    t_y2 = true[:, 3]
    
    for i in range(len(p_x1)):
        p_x1[i] = min(p_x1[i], p_x2[i])
        p_x2[i] = max(p_x1[i], p_x2[i])
        p_y1[i] = min(p_y1[i], p_y2[i])
        p_y2[i] = max(p_y1[i], p_y2[i])
        
    area_t = (t_x2 - t_x1) * (t_y2 - t_y1)
    area_p = (p_x2 - p_x1) * (p_y2 - p_y1)
    
    intersection = torch.zeros(len(p_x1), 4)
    
    for i in range(len(p_x1)):
        intersection[i, 0] = max(p_x1[i], t_x1[i]) #X_1_I
        intersection[i, 1] = max(p_y1[i], t_y1[i]) #Y_1_I
        intersection[i, 2] = min(p_x2[i], t_x2[i]) #X_2_I
        intersection[i, 3] = min(p_y2[i], t_y2[i]) #Y_2_I
    
    I = torch.zeros(len(p_x1))
    
    for i in range(len(p_x1)):
        if intersection[i, 2] > intersection[i, 0] and intersection[i, 3] > intersection[i, 1]:
            I[i] = (intersection[i, 2] - intersection[i, 0]) * (intersection[i, 3] - intersection[i, 1])
        else:
            I[i] = 0
            
    B = torch.zeros(len(p_x1))
