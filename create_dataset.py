import cv2
import numpy as np
import random
import os
import math
import argparse

IMG_SIZE = 256
MIN_DIST = 34

os.makedirs("dataset", exist_ok=True)

# -----------------------------
# Shape generators
# -----------------------------

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def gen_circle():
    r = random.randint(20, 30)

    x = random.randint(r, IMG_SIZE - r)
    y = random.randint(r, IMG_SIZE - r)

    return ("circle", (x, y), r)

def gen_rectangle():
    while True:
        x1 = random.randint(0, IMG_SIZE - 1)
        y1 = random.randint(0, IMG_SIZE - 1)

        x2 = random.randint(0, IMG_SIZE - 1)
        y2 = random.randint(0, IMG_SIZE - 1)

        top_left_x = min(x1, x2)
        top_left_y = min(y1, y2)
        bottom_right_x = max(x1, x2)
        bottom_right_y = max(y1, y2)

        width = bottom_right_x - top_left_x
        height = bottom_right_y - top_left_y

        if width >= MIN_DIST and height >= MIN_DIST:
            return ("rectangle", (top_left_x, top_left_y), (bottom_right_x, bottom_right_y))

def gen_line():
    while True:
        x1 = random.randint(0, IMG_SIZE - 1)
        y1 = random.randint(0, IMG_SIZE - 1)

        x2 = random.randint(0, IMG_SIZE - 1)
        y2 = random.randint(0, IMG_SIZE - 1)

        dx = abs(x1-x2)
        dy = abs(y1-y2)

        if dx >= MIN_DIST or dy >= MIN_DIST:
            if x1 < x2:
                p1 = (x1, y1)
                p2 = (x2, y2)
            else:
                p1 = (x2, y2)
                p2 = (x1, y1)

            return ("line", p1, p2)


generators = [gen_circle, gen_rectangle, gen_line]


# -----------------------------
# Scene generator
# -----------------------------
def get_bounding_box(shape_name, *params, margin=17):
    shape_name = shape_name.lower()
    
    if shape_name == "circle":
        center, radius = params
        x, y = center
        return (x - radius - margin, y - radius - margin, x + radius + margin, y + radius + margin)
    
    elif shape_name == "rectangle":
        pt1, pt2 = params
        x_min = min(pt1[0], pt2[0]) - margin
        y_min = min(pt1[1], pt2[1]) - margin
        x_max = max(pt1[0], pt2[0]) + margin
        y_max = max(pt1[1], pt2[1]) + margin
        return (x_min, y_min, x_max, y_max)
    
    elif shape_name == "line":
        pt1, pt2 = params
        x_min = min(pt1[0], pt2[0]) - margin
        y_min = min(pt1[1], pt2[1]) - margin
        x_max = max(pt1[0], pt2[0]) + margin
        y_max = max(pt1[1], pt2[1]) + margin
        return (x_min, y_min, x_max, y_max)
    
    else:
        raise ValueError(f"Unknown shape name: {shape_name}")
    
def boxes_overlap(box1, box2):
    """
    Check if two bounding boxes overlap.
    box = (x_min, y_min, x_max, y_max)
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # If one box is completely to the left/right or above/below the other, no overlap
    if x1_max < x2_min or x2_max < x1_min:
        return False
    if y1_max < y2_min or y2_max < y1_min:
        return False
    return True

def generate_scene(max_attempts_per_shape=50):
    """
    Generate a scene with 3-5 non-overlapping shapes.
    """
    n_shapes = random.randint(3, 5)
    scene = []
    existing_boxes = []

    attempts = 0
    while len(scene) < n_shapes and attempts < max_attempts_per_shape * n_shapes:
        shape = random.choice(generators)()  # assumes generators() returns a shape tuple
        bbox = get_bounding_box(shape[0], *shape[1:])

        # Check overlap with all existing shapes
        overlap = any(boxes_overlap(bbox, existing_box) for existing_box in existing_boxes)

        if not overlap:
            scene.append(shape)
            existing_boxes.append(bbox)

        attempts += 1  # prevent infinite loops if space is tight

    return scene


# -----------------------------
# Renderer
# -----------------------------

def add_noise(img, mean=0, sigma=10):
    # Generate Gaussian noise
    noise = np.random.normal(mean, sigma, img.shape).astype(np.float32)
    
    # Add noise to image and clip values to valid range [0, 255]
    noisy_img = img.astype(np.float32) + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    
    return noisy_img

def render_scene(scene, name, img_size=(256, 256)):
    # White background
    img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 255

    # Prepare paths
    images_dir = "dataset/images"
    labels_dir = "dataset/labels"
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Image path
    img_path = os.path.join(images_dir, os.path.basename(name))
    # Text path
    txt_name = os.path.splitext(os.path.basename(name))[0] + ".txt"
    txt_path = os.path.join(labels_dir, txt_name)

    # Open text file for writing commands
    with open(txt_path, 'w') as f:
        for shape in scene:
            if not shape:
                continue
            shape_type = shape[0].lower()
            
            # Draw the shape
            if shape_type == "line":
                cv2.line(img, shape[1], shape[2], (0,0,0), 1)
            elif shape_type == "circle":
                cv2.circle(img, shape[1], shape[2], (0,0,0), 1)
            elif shape_type == "rectangle":
                cv2.rectangle(img, shape[1], shape[2], (0,0,0), 1)
            else:
                print(f"Unknown shape: {shape_type}")
                continue
            # Write command to text file
            tokens = []
            for item in shape[1:]:
                if isinstance(item, tuple):
                    tokens.extend(map(str, item))
                else:
                    tokens.append(str(item))
            f.write(f"{shape_type},{','.join(tokens)}\n")
        
        # Write STOP at the end
        f.write("STOP")

    # Save the image
    img = add_noise(img)  # optional noise
    cv2.imwrite(img_path, img)
    print(f"Saved image as {img_path} and commands as {txt_path}")
    


# -----------------------------
# Dataset generator
# -----------------------------

def generate_dataset(n_images):

    os.makedirs("dataset", exist_ok=True)

    for i in range(n_images):

        scene = generate_scene()

        name = f"dataset/img_{i}.png"

        render_scene(scene, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=10, help="Size of dataset")
    args = parser.parse_args()
    generate_dataset(args.size)