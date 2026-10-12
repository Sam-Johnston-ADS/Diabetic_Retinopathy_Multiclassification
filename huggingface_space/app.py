"""
Diabetic Retinopathy Classification & Explainability — Gradio Pipeline
=======================================================================
EfficientNet-B3 + MSAG + BiomedCLIP ensemble with LayerCAM explainability
and Clinical Decision Support System.
Native Gradio interface designed for Hugging Face Spaces.
"""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ============================================================
# CONSTANTS & CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
NUM_CLASSES = 5
INPUT_SIZE = 512

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BIOMEDCLIP_MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

DR_TEXT_PROMPTS = [
    "a color fundus photograph of a normal healthy retina with no diabetic retinopathy, clear macula, normal optic disc, and intact vasculature without any microaneurysms or hemorrhages",
    "a color fundus photograph showing mild nonproliferative diabetic retinopathy with a small number of microaneurysms only",
    "a color fundus photograph showing moderate nonproliferative diabetic retinopathy with microaneurysms, dot and blot hemorrhages, hard exudates, and possible cotton wool spots",
    "a color fundus photograph showing severe nonproliferative diabetic retinopathy with extensive hemorrhages in all four quadrants, venous beading in two or more quadrants, and prominent intraretinal microvascular abnormalities",
    "a color fundus photograph showing proliferative diabetic retinopathy with neovascularization, preretinal or vitreous hemorrhage, and fibrovascular proliferation",
]

DEFAULT_VLM_TEMPERATURE = 2.0
DEFAULT_MSAG_WEIGHT = 0.7

ATTENTION_REDUCTION = 8
DROPOUT_ORIGINAL = 0.40
DROPOUT_SEVERITY = 0.20
DROP_PATH_RATE = 0.10

GRAHAM_SIGMA_X = 30.0

SEVERITY_DESCRIPTIONS = {
    0: "no signs of diabetic retinopathy. The retina appears healthy with no visible microaneurysms, hemorrhages, or exudates.",
    1: "mild nonproliferative diabetic retinopathy (NPDR). Early signs include a few microaneurysms in the retinal vasculature.",
    2: "moderate nonproliferative diabetic retinopathy (NPDR). Findings include scattered hemorrhages, hard exudates, and cotton-wool spots.",
    3: "severe nonproliferative diabetic retinopathy (NPDR). Extensive hemorrhages, venous beading, and intraretinal microvascular abnormalities (IRMA) are observed.",
    4: "proliferative diabetic retinopathy (PDR). Neovascularization is present, indicating advanced disease with risk of vitreous hemorrhage and retinal detachment.",
}

LAYERCAM_DESCRIPTIONS = {
    0: "The LayerCAM signal is diffuse and mostly emphasizes normal anatomical structures (notably the optic disc and major vessels) without strong focal hotspots, consistent with a high-confidence \"No DR\" prediction.",
    1: "The LayerCAM signal highlights small focal regions, likely corresponding to microaneurysms. The attention is relatively scattered with mild concentration near the macula and vascular arcades.",
    2: "The LayerCAM heatmap shows moderate focal activation in regions corresponding to hemorrhages and hard exudates. The model attends to multiple lesion clusters across the retina.",
    3: "The LayerCAM activation is intense and widespread, highlighting extensive hemorrhagic regions, venous beading zones, and areas of intraretinal microvascular abnormalities (IRMA) across multiple quadrants.",
    4: "The LayerCAM signal shows strong activation around neovascularization sites and areas of fibrovascular proliferation. The model focuses on regions at high risk for vitreous hemorrhage and tractional detachment.",
}

CLINICAL_ADVICE = {
    0: {
        "risk_level": "Low",
        "color": "#28a745",
        "follow_up": "Annual screening",
        "referral": "No referral needed. Continue regular diabetic eye screening.",
        "advice": [
            "Schedule next diabetic eye screening in 12 months.",
            "Maintain HbA1c below 7.0% with proper glycemic management.",
            "Keep blood pressure below 140/90 mmHg.",
            "Maintain healthy lipid profile (LDL < 100 mg/dL).",
            "Engage in regular physical activity (at least 150 min/week).",
            "Follow a balanced diet rich in leafy greens and omega-3 fatty acids.",
            "Avoid smoking and limit alcohol consumption.",
            "Report any sudden vision changes to your doctor immediately.",
        ],
    },
    1: {
        "risk_level": "Mild",
        "color": "#ffc107",
        "follow_up": "Re-examine in 6-12 months",
        "referral": "Routine ophthalmology referral recommended.",
        "advice": [
            "Schedule follow-up eye examination in 6-12 months.",
            "Optimize glycemic control — target HbA1c below 7.0%.",
            "Monitor and control blood pressure (target < 130/80 mmHg).",
            "Manage lipid levels with statins if indicated.",
            "No ophthalmic treatment is required at this stage.",
            "Educate patient on DR progression risk factors.",
            "Encourage adherence to diabetes medication regimen.",
            "Report any new floaters, blurred vision, or visual disturbances.",
        ],
    },
    2: {
        "risk_level": "Moderate",
        "color": "#fd7e14",
        "follow_up": "Re-examine in 3-6 months",
        "referral": "Ophthalmology referral recommended. Consider fluorescein angiography.",
        "advice": [
            "Schedule follow-up examination in 3-6 months.",
            "Refer to ophthalmologist for comprehensive dilated eye exam.",
            "Consider fluorescein angiography to assess macular involvement.",
            "Screen for diabetic macular edema (DME) with OCT if available.",
            "Strict glycemic control — target HbA1c below 7.0%.",
            "Aggressive blood pressure management (target < 130/80 mmHg).",
            "Lipid management with statin therapy.",
            "Counsel patient on importance of treatment adherence to slow progression.",
        ],
    },
    3: {
        "risk_level": "High",
        "color": "#dc3545",
        "follow_up": "Re-examine in 2-4 months",
        "referral": "Urgent referral to retina specialist.",
        "advice": [
            "URGENT: Refer to retina specialist within 2-4 weeks.",
            "High risk of progression to proliferative DR (~50% within 1 year).",
            "Consider early panretinal photocoagulation (PRP).",
            "Evaluate for diabetic macular edema — treat with anti-VEGF if present.",
            "Intensive glycemic control — HbA1c target < 7.0%.",
            "Aggressive blood pressure control (target < 130/80 mmHg).",
            "Frequent monitoring with OCT and fundus photography.",
            "Advise patient to report any sudden vision loss, floaters, or flashes immediately.",
        ],
    },
    4: {
        "risk_level": "Very High — Sight-Threatening",
        "color": "#721c24",
        "follow_up": "Re-examine in 1-3 months",
        "referral": "IMMEDIATE referral to retina specialist.",
        "advice": [
            "IMMEDIATE referral to retina specialist — sight-threatening condition.",
            "Panretinal photocoagulation (PRP) is indicated.",
            "Anti-VEGF intravitreal injections may be required.",
            "Risk of vitreous hemorrhage and tractional retinal detachment.",
            "Vitrectomy may be necessary if complications are present.",
            "Follow-up every 1-3 months with retina specialist.",
            "Urgent optimization of systemic glycemic and blood pressure control.",
            "Patient must seek emergency care for sudden vision loss or new floaters.",
        ],
    },
}

# ============================================================
# CHECKPOINT RESOLUTION
# ============================================================

def resolve_checkpoint_path() -> Path:
    candidates = [
        PROJECT_ROOT / "best_model_qwk.pth",
        PROJECT_ROOT / "runs_EfficientNetB3_MSAG_Refined_Stable" / "outputs" / "best_model_qwk.pth",
        PROJECT_ROOT.parent / "runs_EfficientNetB3_MSAG_Refined_Stable" / "outputs" / "best_model_qwk.pth",
        PROJECT_ROOT / "weights" / "best_model_qwk.pth",
        PROJECT_ROOT.parent / "weights" / "best_model_qwk.pth",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    raise FileNotFoundError(
        f"MSAG model checkpoint not found. Checked locations:\n"
        + "\n".join(str(c) for c in candidates)
    )

# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def detect_retinal_circle(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    threshold = max(7, int(np.percentile(gray, 8)))
    binary = (gray > threshold).astype(np.uint8) * 255
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, morph_kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, morph_kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cx, cy, radius = w // 2, h // 2, min(h, w) // 2 - 2
    if contours:
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) > 0.12 * h * w:
            (fcx, fcy), fr = cv2.minEnclosingCircle(contour)
            candidate_radius = fr * 0.98
            if candidate_radius >= 0.28 * min(h, w):
                cx, cy, radius = int(round(fcx)), int(round(fcy)), int(round(candidate_radius))
    radius = max(2, min(radius, cx, cy, w - 1 - cx, h - 1 - cy))
    x1, y1 = max(0, cx - radius), max(0, cy - radius)
    x2, y2 = min(w, cx + radius + 1), min(h, cy + radius + 1)
    crop = rgb[y1:y2, x1:x2].copy()
    local_cx, local_cy = cx - x1, cy - y1
    local_r = min(local_cx, local_cy, crop.shape[1] - 1 - local_cx, crop.shape[0] - 1 - local_cy)
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (local_cx, local_cy), max(1, int(local_r * 0.99)), 255, -1)
    crop[mask == 0] = 0
    return crop, mask, (local_cx, local_cy, local_r)


def graham_normalize(rgb: np.ndarray, mask: np.ndarray, sigma_x: float = GRAHAM_SIGMA_X) -> np.ndarray:
    blurred = cv2.GaussianBlur(rgb, ksize=(0, 0), sigmaX=sigma_x)
    normalized = cv2.addWeighted(rgb, 4.0, blurred, -4.0, 128.0)
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(normalized, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    normalized[mask == 0] = 0
    return normalized


def resize_rgb_mask(rgb: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    out_rgb = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    out_mask = cv2.resize(mask, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_NEAREST)
    out_mask = ((out_mask > 127).astype(np.uint8) * 255)
    out_rgb[out_mask == 0] = 0
    return out_rgb, out_mask


def generate_vessel_map(processed_rgb: np.ndarray, retinal_mask: np.ndarray) -> Tuple[np.ndarray, float]:
    green = processed_rgb[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)
    vessels = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4,
    )
    vessels = cv2.bitwise_and(vessels, retinal_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    vessels = cv2.morphologyEx(vessels, cv2.MORPH_OPEN, kernel)
    vessels = cv2.morphologyEx(vessels, cv2.MORPH_CLOSE, kernel)
    vessel_pixels = np.count_nonzero(vessels)
    density = vessel_pixels / max(1, np.count_nonzero(retinal_mask))
    return vessels, float(density)


def build_explainability_data(rgb: np.ndarray) -> Dict:
    circular, circle_mask, _ = detect_retinal_circle(rgb)
    graham = graham_normalize(circular, circle_mask)
    processed, processed_mask = resize_rgb_mask(graham, circle_mask)
    vessel_map, vessel_density = generate_vessel_map(processed, processed_mask)
    return {
        "graham_rgb": processed,
        "retinal_mask": processed_mask,
        "vessel_map": vessel_map,
        "vessel_density": vessel_density,
    }


def create_vessel_overlay(graham_rgb: np.ndarray, vessel_map: np.ndarray) -> np.ndarray:
    display = graham_rgb.copy()
    vessel_colored = np.zeros_like(display)
    vessel_mask = vessel_map > 0
    vessel_colored[vessel_mask] = [180, 0, 180]
    blended = cv2.addWeighted(display, 0.7, vessel_colored, 0.3, 0)
    blended[~vessel_mask] = display[~vessel_mask]
    return blended

# ============================================================
# MODEL DEFINITIONS
# ============================================================

class MultiScaleSpatialAttentionGate(nn.Module):
    def __init__(self, kernel_sizes: Sequence[int] = (3, 5, 7)):
        super().__init__()
        self.branches = nn.ModuleList(
            [nn.Conv2d(2, 1, kernel_size=k, padding=k // 2, bias=False) for k in kernel_sizes]
        )
        self.bn = nn.BatchNorm2d(len(kernel_sizes))
        self.fuse = nn.Conv2d(len(kernel_sizes), 1, kernel_size=1, bias=True)
        self.last_gate: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.amax(x, dim=1, keepdim=True)
        pooled = torch.cat([avg_map, max_map], dim=1)
        branch_outputs = [branch(pooled) for branch in self.branches]
        multi_scale = torch.cat(branch_outputs, dim=1)
        multi_scale = self.bn(multi_scale)
        gate = torch.sigmoid(self.fuse(multi_scale))
        self.last_gate = gate.detach()
        return x * gate


class RefinedDRModel(nn.Module):
    def __init__(self, drop_path_rate: float = DROP_PATH_RATE):
        super().__init__()
        import timm
        try:
            self.backbone = timm.create_model(
                "efficientnet_b3", pretrained=False, num_classes=0,
                drop_path_rate=drop_path_rate,
            )
        except TypeError:
            self.backbone = timm.create_model(
                "efficientnet_b3", pretrained=False, num_classes=0,
            )
        self.feature_dim = int(self.backbone.num_features)
        self.msag = MultiScaleSpatialAttentionGate()
        self.msag_alpha_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        hidden_att = self.feature_dim // ATTENTION_REDUCTION
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(self.feature_dim, hidden_att), nn.ReLU(inplace=True),
            nn.Linear(hidden_att, self.feature_dim), nn.Sigmoid(),
        )
        self.feature_norm = nn.BatchNorm1d(self.feature_dim)
        self.dropout = nn.Dropout(DROPOUT_ORIGINAL)
        self.severity_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2), nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_SEVERITY), nn.Linear(self.feature_dim // 2, NUM_CLASSES),
        )
        self.lesion_detector = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 4), nn.ReLU(inplace=True),
            nn.Dropout(0.20), nn.Linear(self.feature_dim // 4, 5),
        )
        self.region_predictor = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 4), nn.ReLU(inplace=True),
            nn.Dropout(0.20), nn.Linear(self.feature_dim // 4, 5),
        )
        self.last_feature_map: Optional[torch.Tensor] = None
        self.fused_identity = nn.Identity()

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.backbone.forward_features(x)
        self.last_feature_map = features
        msag_features = self.msag(features)
        msag_alpha = 0.25 * torch.tanh(self.msag_alpha_raw)
        fused = self.fused_identity(features + msag_alpha * (msag_features - features))
        pooled = F.adaptive_avg_pool2d(fused, 1).flatten(1)
        pooled_4d = pooled.unsqueeze(-1).unsqueeze(-1)
        attention_weights = self.attention(pooled_4d)
        attended = pooled * attention_weights
        normalized = self.feature_norm(attended)
        normalized = self.dropout(normalized)
        severity_logits = self.severity_classifier(normalized)
        if return_features:
            return {
                "severity": severity_logits,
                "lesions": self.lesion_detector(normalized),
                "regions": self.region_predictor(normalized),
                "features": normalized, "feature_map": fused,
            }
        return severity_logits

# ============================================================
# EXPLAINABILITY (LayerCAM)
# ============================================================

class LayerCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._fwd_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None):
        self.model.zero_grad()
        output = self.model(x)
        if class_idx is None:
            class_idx = torch.argmax(output).item()
        score = output[0, class_idx]
        score.backward()
        positive_gradients = torch.relu(self.gradients)
        cam = torch.sum(positive_gradients * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-7)
        return cam.cpu().detach().numpy()[0, 0], class_idx, torch.softmax(output, dim=1)

    def close(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def prepare_heatmap(heatmap: np.ndarray, target_h: int, target_w: int,
                    mask: Optional[np.ndarray] = None) -> np.ndarray:
    heatmap_resized = cv2.resize(heatmap, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    if mask is not None:
        mask_bool = mask > 0
        heatmap_resized[~mask_bool] = 0.0
        vals = heatmap_resized[mask_bool]
    else:
        vals = heatmap_resized.ravel()

    vmin = float(vals.min()) if vals.size > 0 else 0.0
    vmax = float(vals.max()) if vals.size > 0 else 0.0
    if vmax - vmin > 1e-8:
        heatmap_resized = (heatmap_resized - vmin) / (vmax - vmin)

    if mask is not None:
        heatmap_resized[~mask_bool] = 0.0

    return np.clip(heatmap_resized, 0.0, 1.0)


def _find_optic_disc_center(gray: np.ndarray, retinal_mask: Optional[np.ndarray] = None) -> Tuple[int, int]:
    h, w = gray.shape[:2]
    if retinal_mask is not None:
        gray_search = gray.copy()
        gray_search[retinal_mask == 0] = 0
        retina_vals = gray[retinal_mask > 0]
    else:
        gray_search = gray
        retina_vals = gray[gray > 10]

    if retina_vals.size < 100:
        return w // 2, h // 2

    od_radius_est = int(min(h, w) * 0.07)
    best_score = -1.0
    best_center: Optional[Tuple[int, int]] = None
    min_od_area = np.pi * (od_radius_est * 0.3) ** 2
    max_od_area = np.pi * (od_radius_est * 5.0) ** 2

    for pct in (90, 93, 95):
        thresh_val = np.percentile(retina_vals, pct)
        bright = (gray_search > thresh_val).astype(np.uint8) * 255
        kern_sz = max(5, od_radius_est)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kern_sz, kern_sz))
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_od_area or area > max_od_area:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter < 1:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            M = cv2.moments(contour)
            if M["m00"] < 1:
                continue
            ccx = int(M["m10"] / M["m00"])
            ccy = int(M["m01"] / M["m00"])
            region_mask = np.zeros_like(gray)
            cv2.drawContours(region_mask, [contour], -1, 255, -1)
            mean_bright = cv2.mean(gray, mask=region_mask)[0] / 255.0
            score = (circularity * 0.5 + mean_bright * 0.3 + min(area / max_od_area, 1.0) * 0.2)
            if score > best_score:
                best_score = score
                best_center = (ccx, ccy)

    if best_center is not None:
        return best_center

    blur_sigma = max(h, w) * 0.15
    blurred = cv2.GaussianBlur(gray_search, (0, 0), blur_sigma)
    if retinal_mask is not None:
        blurred[retinal_mask == 0] = 0
    _, _, _, max_loc = cv2.minMaxLoc(blurred)
    return max_loc


def suppress_optic_disc(heatmap: np.ndarray, rgb: np.ndarray,
                        retinal_mask: Optional[np.ndarray] = None) -> np.ndarray:
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cx, cy = _find_optic_disc_center(gray, retinal_mask)
    radius_hard = int(min(h, w) * 0.14)
    radius_soft = int(min(h, w) * 0.23)
    suppression = np.ones((h, w), dtype=np.float32)
    cv2.circle(suppression, (cx, cy), radius_hard, 0, -1)
    fade_sigma = (radius_soft - radius_hard) * 0.7
    ksize = int(np.ceil(fade_sigma * 6)) | 1
    suppression = cv2.GaussianBlur(suppression, (ksize, ksize), fade_sigma)
    suppression = np.clip(suppression, 0.0, 1.0)
    return heatmap * suppression

# ============================================================
# PIPELINE INITIALIZATION & CACHING
# ============================================================

def _normalize_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        state = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint.get("model")
        )
    else:
        state = checkpoint
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint does not contain a recognizable state_dict.")
    normalized = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        clean_key = key
        for prefix in ("module.", "model.", "_orig_mod."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
        normalized[clean_key] = value
    return normalized


@lru_cache(maxsize=1)
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load MSAG model
    msag_model = RefinedDRModel()
    ckpt_path = resolve_checkpoint_path()
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = _normalize_state_dict(checkpoint)
    msag_model.load_state_dict(state_dict, strict=False)
    msag_model.to(device)
    msag_model.eval()

    # 2. Load BiomedCLIP model
    import open_clip
    vlm_model, _, vlm_preprocess = open_clip.create_model_and_transforms(BIOMEDCLIP_MODEL_NAME)
    vlm_tokenizer = open_clip.get_tokenizer(BIOMEDCLIP_MODEL_NAME)
    vlm_model.to(device)
    vlm_model.eval()
    for param in vlm_model.parameters():
        param.requires_grad = False

    # 3. Compute VLM Text Embeddings
    tokens = vlm_tokenizer(DR_TEXT_PROMPTS).to(device)
    with torch.no_grad():
        text_features = vlm_model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)

    return msag_model, vlm_model, vlm_preprocess, text_features, device


MSAG_TRANSFORM = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ============================================================
# VISUAL REPORT BUILDER
# ============================================================

def build_report_figure(original_image: Image.Image, results: Dict) -> plt.Figure:
    msag_pred = results["msag_pred"]
    msag_conf = results["msag_probs"][msag_pred] * 100
    msag_probs = results["msag_probs"]
    vessel_overlay = results["vessel_overlay"]
    vessel_density = results["vessel_density"]
    heatmap = results["heatmap"]
    input_rgb = results["input_rgb"]

    fig = plt.figure(figsize=(18, 10), facecolor="#1a1a2e")
    gs = gridspec.GridSpec(
        2, 3,
        height_ratios=[1.2, 0.8],
        hspace=0.25, wspace=0.15,
        left=0.03, right=0.97, top=0.92, bottom=0.08,
    )

    # Panel 1: Original Input
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(np.array(original_image.convert("RGB")))
    ax_orig.set_title("Original Fundus Photograph", fontsize=13, fontweight="bold", pad=8, color="#FFA500")
    ax_orig.axis("off")

    # Panel 2: Processed + Vessels
    ax_vessel = fig.add_subplot(gs[0, 1])
    ax_vessel.imshow(vessel_overlay)
    ax_vessel.set_title(
        f"Processed Fundus + Retinal Vessels\nVessel Density: {vessel_density:.4f}",
        fontsize=12, fontweight="bold", pad=8, color="#FFA500",
    )
    ax_vessel.axis("off")

    # Panel 3: LayerCAM Heatmap
    ax_cam = fig.add_subplot(gs[0, 2])
    orig_rgb = np.array(original_image.convert("RGB"))
    orig_h, orig_w = orig_rgb.shape[:2]
    orig_gray = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY)
    _, orig_mask = cv2.threshold(orig_gray, 15, 255, cv2.THRESH_BINARY)
    k_orig = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    orig_mask = cv2.morphologyEx(orig_mask, cv2.MORPH_CLOSE, k_orig)
    orig_mask = cv2.morphologyEx(orig_mask, cv2.MORPH_OPEN, k_orig)
    heatmap_norm = prepare_heatmap(heatmap, orig_h, orig_w, mask=orig_mask)
    heatmap_norm = suppress_optic_disc(heatmap_norm, orig_rgb, retinal_mask=orig_mask)

    mask_bool = orig_mask > 0
    base_gray = np.stack([orig_gray, orig_gray, orig_gray], axis=-1)

    heatmap_clipped = np.clip(heatmap_norm, 0, 1)
    heatmap_uint8 = np.uint8(255 * heatmap_clipped)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_PLASMA)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(base_gray, 0.4, colored_heatmap, 0.6, 0)
    overlay[~mask_bool] = 0

    ax_cam.imshow(overlay)
    ax_cam.set_title(
        f"LayerCAM (Optic Disc Suppressed)\nPrediction: {CLASS_NAMES[msag_pred]} ({msag_conf:.1f}%)",
        fontsize=12, fontweight="bold", color="#FFA500", pad=8,
    )
    ax_cam.axis("off")
    sm = ScalarMappable(cmap="plasma", norm=Normalize(0, 1))
    sm.set_array([])
    divider_cam = make_axes_locatable(ax_cam)
    cax_cam = divider_cam.append_axes("right", size="5%", pad=0.05)
    cb_cam = fig.colorbar(sm, cax=cax_cam)
    cb_cam.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb_cam.ax.yaxis.get_ticklabels(), color="white")

    # Panel 4: Confidence Bar Chart
    ax_bar = fig.add_subplot(gs[1, :])
    x_pos = np.arange(NUM_CLASSES)
    bar_colors = [CLINICAL_ADVICE[i]["color"] if i == msag_pred else "#DDA0DD" for i in range(NUM_CLASSES)]

    bars = ax_bar.bar(x_pos, msag_probs, color=bar_colors, width=0.55, edgecolor="white", linewidth=1.5)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(CLASS_NAMES, fontsize=11, color="#cccccc")
    ax_bar.set_ylabel("Probability", fontsize=11, color="#cccccc")
    ax_bar.set_title("Severity Probability Distribution (MSAG Primary)", fontsize=13, fontweight="bold", pad=10, color="white")
    ax_bar.set_ylim(0, 1.08)
    ax_bar.set_facecolor("#16213e")
    for spine in ax_bar.spines.values():
        spine.set_color("#cccccc")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.tick_params(colors="#cccccc")

    for bar, prob in zip(bars, msag_probs):
        pct = prob * 100
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{pct:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold",
            color="white",
        )

    return fig

# ============================================================
# MAIN INFERENCE FUNCTION
# ============================================================

def predict_fundus(
    image: Optional[Image.Image],
    msag_weight: float = DEFAULT_MSAG_WEIGHT,
    vlm_temperature: float = DEFAULT_VLM_TEMPERATURE,
):
    if image is None:
        raise gr.Error("Please upload a retinal fundus image or click one of the examples below.")

    msag_model, vlm_model, vlm_preprocess, text_features, device = load_models()

    image_rgb = image.convert("RGB")
    rgb_np = np.array(image_rgb)

    # 1. Explainability preprocessing
    explain = build_explainability_data(rgb_np)
    vessel_overlay = create_vessel_overlay(explain["graham_rgb"], explain["vessel_map"])

    # 2. MSAG forward pass
    msag_input = MSAG_TRANSFORM(image_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        msag_logits = msag_model(msag_input)
        msag_probs = F.softmax(msag_logits, dim=-1).cpu().numpy()[0]

    # 3. BiomedCLIP forward pass
    vlm_weight = round(1.0 - msag_weight, 2)
    vlm_input = vlm_preprocess(image_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = vlm_model.encode_image(vlm_input)
        image_features = F.normalize(image_features, dim=-1)
        logit_scale = vlm_model.logit_scale.exp()
        logits = logit_scale * (image_features @ text_features.t())
        vlm_probs = F.softmax(logits / vlm_temperature, dim=-1).cpu().numpy()[0]

    # 4. LayerCAM
    cam = LayerCAM(msag_model, msag_model.backbone.conv_head)
    try:
        heatmap, cam_pred, _ = cam(msag_input, class_idx=None)
    finally:
        cam.close()
    msag_model.eval()

    # 5. Ensemble aggregation
    ensemble_probs = msag_weight * msag_probs + vlm_weight * vlm_probs
    msag_pred = int(np.argmax(msag_probs))
    vlm_pred = int(np.argmax(vlm_probs))
    ensemble_pred = int(np.argmax(ensemble_probs))

    input_rgb_np = np.array(image_rgb.resize((INPUT_SIZE, INPUT_SIZE)))

    results = {
        "msag_probs": msag_probs,
        "vlm_probs": vlm_probs,
        "ensemble_probs": ensemble_probs,
        "msag_pred": msag_pred,
        "vlm_pred": vlm_pred,
        "ensemble_pred": ensemble_pred,
        "heatmap": heatmap,
        "input_rgb": input_rgb_np,
        "vessel_overlay": vessel_overlay,
        "vessel_density": explain["vessel_density"],
    }

    # Generate Matplotlib Visual Report
    report_fig = build_report_figure(image, results)

    # Format Gradio Probabilities dictionary
    prob_dict = {CLASS_NAMES[i]: float(ensemble_probs[i]) for i in range(NUM_CLASSES)}

    # Clinical details
    advice = CLINICAL_ADVICE[msag_pred]
    finding_text = SEVERITY_DESCRIPTIONS[msag_pred]
    layercam_text = LAYERCAM_DESCRIPTIONS[msag_pred]
    msag_conf = msag_probs[msag_pred] * 100
    ensemble_conf = ensemble_probs[ensemble_pred] * 100
    models_agree = (msag_pred == vlm_pred)

    # HTML Banner for Diagnostic Consensus
    if models_agree:
        agreement_html = f"""
        <div style="background:#e8f5e9;border-left:5px solid #2e7d32;padding:12px;border-radius:6px;margin-bottom:12px;color:#1b5e20;">
            <strong>✓ Multi-Model Consensus:</strong> Both MSAG and BiomedCLIP agree on 
            <strong>{CLASS_NAMES[msag_pred]}</strong> (Ensemble: {ensemble_conf:.1f}%).
        </div>
        """
    else:
        agreement_html = f"""
        <div style="background:#fff8e1;border-left:5px solid #ffa000;padding:12px;border-radius:6px;margin-bottom:12px;color:#e65100;">
            <strong>⚠️ Model Discrepancy:</strong> MSAG primary model predicts <strong>{CLASS_NAMES[msag_pred]}</strong> ({msag_conf:.1f}%), 
            while BiomedCLIP predicts <strong>{CLASS_NAMES[vlm_pred]}</strong> ({vlm_probs[vlm_pred]*100:.1f}%). 
            Ensemble consensus favors <strong>{CLASS_NAMES[ensemble_pred]}</strong>.
        </div>
        """

    diagnostic_card_html = f"""
    <div style="background:#ffffff;border:1px solid #e0e0e0;border-radius:10px;padding:16px;box-shadow:0 2px 4px rgba(0,0,0,0.05);">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">
            <span style="background-color:{advice['color']};color:white;font-weight:700;font-size:1.15rem;padding:6px 14px;border-radius:6px;">
                {CLASS_NAMES[msag_pred]} &bull; {msag_conf:.1f}%
            </span>
            <span style="font-size:0.9rem;font-weight:600;color:#555;">
                Risk Level: <strong style="color:{advice['color']};">{advice['risk_level']}</strong>
            </span>
        </div>
        {agreement_html}
        <div style="font-size:0.9rem;color:#444;">
            <strong>Follow-up Recommendation:</strong> {advice['follow_up']}
        </div>
    </div>
    """

    # Format Markdown Report
    advice_bullets = "\n".join(
        f"- {'🔴 **URGENT:** ' + item.replace('URGENT: ', '') if item.startswith('URGENT') else ('🔴 **IMMEDIATE:** ' + item.replace('IMMEDIATE ', '') if item.startswith('IMMEDIATE') else item)}"
        for item in advice["advice"]
    )

    clinical_report_md = f"""
### 🔬 Clinical Findings
> **Lesion Pathology:** This retinal fundus image shows {finding_text}

> **Vessel Density Metric:** Estimated retinal vasculature density is **{explain['vessel_density']:.4f}**.

> **Visual Attention (LayerCAM):** {layercam_text}

---

### 📋 Doctor's Recommendations & Guidance
- **Referral Plan:** {advice['referral']}
- **Follow-up Interval:** {advice['follow_up']}

**Actionable Patient Guidance:**
{advice_bullets}

---
> ⚠️ **Clinical Disclaimer:** This application is an AI-assisted diagnostic research tool for educational and screening assistance. It does not replace professional medical diagnosis. All clinical decisions must be confirmed by a licensed ophthalmologist or retina specialist.
    """

    return diagnostic_card_html, prob_dict, report_fig, clinical_report_md

# ============================================================
# GRADIO BLOCKS INTERFACE
# ============================================================

custom_css = """
#main-header { text-align: center; margin-bottom: 20px; }
#main-header h1 { color: #1a237e; font-weight: 800; font-size: 2.2rem; margin-bottom: 4px; }
#main-header p { color: #5c6bc0; font-size: 1.05rem; }
.card { border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); padding: 12px; }
"""

with gr.Blocks(title="Diabetic Retinopathy Classification & Explainability") as demo:
    gr.HTML(
        """
        <div id="main-header">
            <h1>👁️ Diabetic Retinopathy Multiclassification & Explainability</h1>
            <p><strong>EfficientNet-B3 + MSAG Attention &bull; Microsoft BiomedCLIP VLM &bull; LayerCAM &bull; Vessel Segmentation</strong></p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=4):
            image_input = gr.Image(
                type="pil",
                label="Upload Retinal Fundus Image",
                sources=["upload", "clipboard"],
            )

            with gr.Accordion("⚙️ Ensemble & Inference Settings", open=False):
                msag_weight_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.05,
                    label="MSAG Model Weight (Primary CNN)",
                    info="Weight for the trained EfficientNet-B3 + MSAG model (remaining weight goes to BiomedCLIP).",
                )
                vlm_temp_slider = gr.Slider(
                    minimum=0.5,
                    maximum=5.0,
                    value=2.0,
                    step=0.25,
                    label="BiomedCLIP Temperature",
                    info="Softmax temperature for zero-shot text matching. Higher = flatter distribution.",
                )

            analyze_btn = gr.Button("🔍 Run Diagnostic Analysis", variant="primary", size="lg")

            # Example Fundus Images
            example_paths = [
                str(p) for p in sorted((PROJECT_ROOT / "examples").glob("*.png"))
            ]
            if not example_paths:
                example_paths = [
                    str(p) for p in sorted((PROJECT_ROOT / "examples").glob("*.*"))
                ]

            if example_paths:
                gr.Examples(
                    examples=[[p, 0.7, 2.0] for p in example_paths[:4]],
                    inputs=[image_input, msag_weight_slider, vlm_temp_slider],
                    label="Sample Retinal Fundus Cases",
                )

        with gr.Column(scale=6):
            diagnostic_banner = gr.HTML(label="Diagnostic Assessment")
            probs_label = gr.Label(num_top_classes=5, label="Severity Probability Distribution")
            visual_report = gr.Plot(label="Diagnostic & Explainability Visual Report")
            clinical_report = gr.Markdown(label="Clinical Findings & Recommendations")

    analyze_btn.click(
        fn=predict_fundus,
        inputs=[image_input, msag_weight_slider, vlm_temp_slider],
        outputs=[diagnostic_banner, probs_label, visual_report, clinical_report],
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
