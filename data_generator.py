from random import randint
import cv2
import math

def generator_images(im_size, min_ants, max_ants, body_radius, head_radius, image_path):
    color = (0,0,0)
    number_of_ants = randint(min_ants, max_ants)
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    for i in range(number_of_ants):
        x_center_body = randint(body_radius + 2 * head_radius, im_size[0] - (body_radius + 2 * head_radius))
        y_center_body = randint(body_radius + 2 * head_radius, im_size[0] - (body_radius + 2 * head_radius))
        img = cv2.circle(img, (x_center_body, y_center_body), body_radius, color, -1)
        
        flag = False
        while not flag:
            x_crossing = randint(body_radius + 2 * head_radius, im_size[0] - (body_radius + 2 * head_radius))
            y_crossing = randint(body_radius + 2 * head_radius, im_size[0] - (body_radius + 2 * head_radius))
            
            if (x_crossing - x_center_body)**2 + (y_crossing - y_center_body)**2 == body_radius**2:
                flag = True
        
    
