import torch
import numpy as np
import cv2

def render_shape(shape_name, params, canvas_size=(256, 256), device='cpu'):

    # Create blank white canvas
    img = np.ones(canvas_size, dtype=np.uint8) * 255  # white background    
    # Draw shape
    if shape_name == "line":
        if len(params) != 4:
            raise ValueError("Line requires 4 params: x1, y1, x2, y2")
        pt1 = (int(params[0]), int(params[1]))
        pt2 = (int(params[2]), int(params[3]))
        cv2.line(img, pt1, pt2, color=0, thickness=1)  # black line
    elif shape_name == "circle":
        if len(params) != 3:
            raise ValueError("Circle requires 3 params: x_center, y_center, radius")
        center = (int(params[0]), int(params[1]))
        radius = int(params[2])
        cv2.circle(img, center, radius, color=0, thickness=1)
    elif shape_name == "rectangle":
        if len(params) != 4:
            raise ValueError("Rectangle requires 4 params: x1, y1, x2, y2")
        pt1 = (int(params[0]), int(params[1]))
        pt2 = (int(params[2]), int(params[3]))
        cv2.rectangle(img, pt1, pt2, color=0, thickness=1)
    else:
        raise ValueError(f"Unknown shape: {shape_name}")

    # Convert to float tensor in range [0,1], shape (1, H, W)
    gt_shape = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0).to(device)

    return gt_shape