import matplotlib.pyplot as plt
import torch
from model import ShapeReconstructor
from PIL import Image
import torchvision.transforms as transforms
from render import render_shape

IMG_SIZE = 256
MIN_DIST = 34

SHAPE_PARAM_COUNTS = {"circle": 3, "rectangle": 4, "line": 4, "stop": 0}

def map_index_to_shape(idx):
    """Map model output index to shape name."""
    shape_dict = {0: "circle", 1: "rectangle", 2: "line", 3: "stop"}
    return shape_dict.get(idx, "unknown")

def decode_image(model, target, device, max_commands=8, greedy=True):
    """
    Inference function for ShapeReconstructor model
    """
    model.eval()
    canvas = torch.zeros_like(target)
    pred_cmds = []

    for step in range(max_commands):
        # --- Predict shape ---
        out = model(target, canvas, shape_idx=None,
                    prev_param_tokens=torch.zeros(1, 0, dtype=torch.long, device=device),
                    step=0)
        logits = out["cmd_logits"]
        if greedy:
            shape_idx = torch.argmax(logits, dim=-1)
        else:
            prob = torch.softmax(logits, dim=-1)
            shape_idx = torch.multinomial(prob, num_samples=1)
        shape_name = map_index_to_shape(shape_idx.item())
        if shape_name == "stop":
            break

        # --- Predict parameters ---
        num_params = SHAPE_PARAM_COUNTS[shape_name]
        prev_params = torch.zeros(1, 0, dtype=torch.long, device=device)
        params_pixel = []

        for k in range(num_params):
            out = model(target, canvas, shape_idx=shape_idx, prev_param_tokens=prev_params, step=k+1)
            logits = out["param_logits"]

            if greedy:
                param_tok = torch.argmax(logits, dim=-1)
            else:
                prob = torch.softmax(logits, dim=-1)
                param_tok = torch.multinomial(prob, num_samples=1)

            # store discrete token for next step input
            prev_params = torch.cat([prev_params, param_tok.unsqueeze(0)], dim=1)

            # --- Map token (0-15) back to pixel value (0-255) ---
            param_pixel = param_tok.item() / 15 * 255
            params_pixel.append(param_pixel)

        # --- Update canvas with mapped parameters ---
        pred_canvas = render_shape(shape_name, params_pixel, device=device)
        canvas = canvas + pred_canvas

        # --- Save predicted command ---
        pred_cmds.append({"shape": shape_name, "params": [int(p) for p in params_pixel]})

    return pred_cmds, canvas.squeeze().cpu().numpy()

def main(checkpoint_path, img_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ──
    print(f"[inference] loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    saved_args = ckpt.get("args", {})
    img_size   = saved_args.get("img_size", IMG_SIZE)
    model = ShapeReconstructor(img_size=img_size).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[inference] model loaded (val_loss={ckpt.get('val_loss', '?'):.4f}, "
          f"epoch={ckpt.get('epoch', '?')})")

    # ── Load & preprocess your image ──
    transform = transforms.Compose([
        transforms.Grayscale(),              # if your model expects 1 channel
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),               # (C, H, W) in [0,1]
    ])

    img = Image.open(img_path)
    target = transform(img).unsqueeze(0).to(device)   # (1, 1, H, W)

    # ── Run inference ──
    print("\n[1/1] Running inference on input image...")

    pred_cmds, pred_canvas = decode_image(
        model, target, device,
    )

    # ── Print predicted commands ──
    print("Predicted commands:")
    for c in pred_cmds:
        print(f"  {c['shape']:12s} {c['params']}")

    # ── Render predicted canvas ──
    # Convert tensor/canvas to numpy if needed
    pred_canvas_np = pred_canvas  # already returned as numpy from decode_image()

    plt.figure(figsize=(6, 6))
    plt.imshow(pred_canvas_np, cmap='gray')
    plt.title("Predicted Canvas")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    
    checkpoint_path = "/content/checkpoints/best_model.pt"
    img_path = "/content/dataset/images/img_0.png"

    main(checkpoint_path, img_path)