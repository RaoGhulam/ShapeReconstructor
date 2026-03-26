import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────

GRID_SIZE   = 16
NUM_SHAPES  = 4
SHAPE_NAMES = ["circle", "rectangle", "line", "stop"]

# Number of parameter tokens for each shape type
SHAPE_PARAM_COUNTS = {"circle": 3, "rectangle": 4, "line": 4, "stop": 0}
MAX_PARAMS = 4            # maximum number of parameter tokens in any command

CNN_FEATURE_DIM = 1960     # flattened feature map size


# ─────────────────────────────────────────────
#  1.  CNN Feature Extractor
# ─────────────────────────────────────────────

class CNNFeatureExtractor(nn.Module):
    """
    Input:
        Target: [1, 1, 256, 256],
        Canvas [1, 1, 256, 256].
    Output:
        Feature Map: [1, 10, 14, 14]
    """
    def __init__(self):
        super().__init__()

        # Parallel branches
        self.branch1 = nn.Sequential(
            nn.Conv2d(2, 20, kernel_size=8, padding="same"),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=(16, 4), padding="same"),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=(4, 16), padding="same"),
            nn.ReLU(inplace=True)
        )

        # Pooling after concatenation
        self.pool1 = nn.MaxPool2d(kernel_size=8, stride=4)

        # Layer 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(24, 10, kernel_size=8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4)
        )

    def forward(self, target, canvas):
        x = torch.cat([target, canvas], dim=1)

        # Parallel paths
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        x = torch.cat([b1, b2, b3], dim=1)

        x = self.pool1(x)
        x = self.layer2(x)

        return x


# ─────────────────────────────────────────────
#  2.  Command Type Predictor (t1)
# ─────────────────────────────────────────────

class CommandTypePredictor(nn.Module):
    """
    Input:
        Feature Map from CNN: [1, 10, 14, 14]
    Output:
        Logits (B, NUM_SHAPES): [1, 4]
    """

    def __init__(self, in_dim: int = CNN_FEATURE_DIM, num_classes: int = NUM_SHAPES):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        B = f.size(0)
        f_flat = f.view(B, -1)
        logits = self.linear(f_flat)
        return logits


# ─────────────────────────────────────────────
#  3.  Spatial Transformer Network (STN)
# ─────────────────────────────────────────────
class SpatialTransformerNetwork(nn.Module):
    def __init__(self,
        num_shapes: int = NUM_SHAPES,
        prev_token_dim = MAX_PARAMS * GRID_SIZE,
    ):
        super().__init__()

        # Input: concatenation of one-hot previous tokens
        max_onehot_dim = num_shapes + prev_token_dim
        self.localization = nn.Linear(max_onehot_dim, 6)

        # Initialize to identity transform
        self.localization.weight.data.zero_()
        self.localization.bias.data.copy_(torch.tensor([1,0,0,0,1,0], dtype=torch.float))

    def forward(self, f, prev_onehots):
        # Predict affine theta: (B, 2, 3)
        theta = self.localization(prev_onehots).view(-1, 2, 3)

        # Generate sampling grid (matches input size)
        grid = F.affine_grid(theta, size=f.size(), align_corners=False)

        # Apply affine transformation to features
        f_prime = F.grid_sample(f, grid, align_corners=False)

        return f_prime

class SpatialTransformerFactory:
    """Returns the appropriate SpatialTransformerNetwork for a given shape."""
    @staticmethod
    def build(num_shapes: int = NUM_SHAPES, grid_size: int = GRID_SIZE,
              max_prev_tokens=MAX_PARAMS) -> SpatialTransformerNetwork:
        return SpatialTransformerNetwork(
            num_shapes = num_shapes,
            prev_token_dim = max_prev_tokens * grid_size
        )
# ─────────────────────────────────────────────
#  4.  Parameter Token Predictor (t2..tn)
# ─────────────────────────────────────────────

class ParameterTokenPredictor(nn.Module):
    """
    MLP that predicts the next coordinate token (value in [0, GRID_SIZE)).
    For line commands a hidden layer of 32 neurons is used; for others no hidden layer.
    """

    def __init__(
        self,
        feature_dim: int = CNN_FEATURE_DIM,
        grid_size: int = GRID_SIZE,
        num_shapes: int = NUM_SHAPES,
        prev_token_dim = MAX_PARAMS * GRID_SIZE,
        hidden_size: int = 0,   # 0 = no hidden layer
    ):
        super().__init__()
        in_dim = feature_dim + num_shapes + prev_token_dim

        if hidden_size > 0:
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, grid_size),
            )
        else:
            self.mlp = nn.Linear(in_dim, grid_size)

    def forward(
        self,
        f_prime: torch.Tensor,        # (B, feature_dim)  attended feature
        prev_onehots: torch.Tensor,   # (B, num_shapes + max_prev * grid_size)
    ) -> torch.Tensor:
        """Returns logits : (B, GRID_SIZE)"""
        f_flat = f_prime.view(f_prime.size(0), -1)
        x = torch.cat([f_flat, prev_onehots], dim=1)
        return self.mlp(x)


class ShapeHeadFactory:
    """Returns the appropriate ParameterTokenPredictor for a given shape."""
    @staticmethod
    def build(shape: str, feature_dim=CNN_FEATURE_DIM, grid_size=GRID_SIZE,
              num_shapes=NUM_SHAPES, max_prev_tokens=MAX_PARAMS) -> ParameterTokenPredictor:
        hidden = 32 if shape == "line" else 0
        return ParameterTokenPredictor(
            feature_dim = feature_dim,
            grid_size = grid_size,
            num_shapes = num_shapes,
            prev_token_dim = max_prev_tokens * grid_size,
            hidden_size = hidden,
        )


# ─────────────────────────────────────────────
#  5.  Full ShapeReconstructor Model
# ─────────────────────────────────────────────

class ShapeReconstructor(nn.Module):
    """
    Full model that, given a target image and the current canvas,
    produces logits for:
      - command type  (t1)
      - each parameter token (t2..tn)  for a given shape type and step

    During training we use teacher forcing: the ground-truth previous
    tokens are passed in.  During inference we use autoregressive decoding
    (see inference.py).
    """

    def __init__(self, img_size: int = 256):
        super().__init__()
        self.img_size = img_size

        self.cnn = CNNFeatureExtractor()
        self.cmd_predictor = CommandTypePredictor()

        # One SpatialTransformerNetwork per parameter per shape
        self.stn_heads = nn.ModuleDict({
            shape: nn.ModuleList([
                SpatialTransformerFactory.build(
                    max_prev_tokens = i
                )
                for i in range(SHAPE_PARAM_COUNTS[shape])
            ])
            for shape in ["circle", "rectangle", "line"]
        })

        # One parameter-token head per shape (line gets hidden layer per spec)
        self.param_heads = nn.ModuleDict({
            shape: nn.ModuleList([
                ShapeHeadFactory.build(
                    shape,
                    max_prev_tokens = i
                )
                for i in range(SHAPE_PARAM_COUNTS[shape])
            ])
            for shape in ["circle", "rectangle", "line"]
        })

    # ── helpers ──────────────────────────────

    @staticmethod
    def _make_onehot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
        """indices: (B,) → one-hot (B, num_classes)"""
        return F.one_hot(indices.long(), num_classes=num_classes).float()

    # ── forward ──────────────────────────────

    def forward(
        self,
        target: torch.Tensor,          # (B, 1, H, W)
        canvas: torch.Tensor,          # (B, 1, H, W)
        shape_idx: torch.Tensor,       # (B,) ground-truth shape type index (for teacher forcing)
        prev_param_tokens: torch.Tensor,  # (B, n_prev) ground-truth previous param indices
        step: int,                     # which parameter token we are predicting (0-indexed)
    ) -> dict:
        """
        Returns a dict with:
          'cmd_logits'   : (B, NUM_SHAPES)   – always computed
          'param_logits' : (B, GRID_SIZE)    – only when step >= 0
        """
        B = target.shape[0]

        # ── Step 1: CNN feature extraction ──
        f = self.cnn(target, canvas)                      # f is a tensor

        # ── Step 2: Command type logits ──
        if step == 0:
            cmd_logits = self.cmd_predictor(f)
            param_logits = None

        # ── Step 3: Parameter token prediction ──
        else:
            cmd_logits = None

            param_step = step-1

            shape_oh = self._make_onehot(shape_idx, NUM_SHAPES)  # (B, NUM_SHAPES)

            # Encode each previous param token as one-hot(GRID_SIZE) and concat
            if prev_param_tokens.shape[1] > 0:
                param_ohs = torch.cat(
                    [self._make_onehot(prev_param_tokens[:, i], GRID_SIZE)
                    for i in range(prev_param_tokens.shape[1])],
                    dim=1,
                )  # (B, n_prev * GRID_SIZE)
            else:
                param_ohs = torch.zeros(B, 0, device=target.device)

            prev_onehots_raw = torch.cat([shape_oh, param_ohs], dim=1)  # (B, NUM_SHAPES + n_prev*GRID_SIZE)

            shape_name = SHAPE_NAMES[shape_idx[0].item()]

            # STN attention
            stn_head = self.stn_heads[shape_name][param_step]     
            f_prime = stn_head(f, prev_onehots_raw)         # (B, CNN_FEATURE_DIM)

            # param head input = [f' | prev_onehots (padded to head input size)]
            head = self.param_heads[shape_name][param_step]
            param_logits = head(f_prime, prev_onehots_raw)    # (B, GRID_SIZE)

        return {
            "cmd_logits":   cmd_logits,
            "param_logits": param_logits,
            "features":     f,
        }