---
title: Diabetic Retinopathy Multiclassification & Explainability
emoji: 👁️
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: 5-Class Diabetic Retinopathy screening with MSAG attention and LayerCAM explainability.
---

# 👁️ Diabetic Retinopathy Multiclassification & Explainability

An AI-assisted diagnostic screening tool for **5-Class Diabetic Retinopathy (DR)** severity assessment, lesion explainability, and clinical guidance based on the International Clinical Diabetic Retinopathy (ICDR) scale.

---

## 📌 Features
- **Primary Deep Learning Model:** EfficientNet-B3 with custom **Multi-Scale Spatial Attention Gate (MSAG)** capturing micro-lesions at $3\times3$, $5\times5$, and $7\times7$ spatial scales, with learnable residual feature fusion.
- **Vision-Language Ensemble:** Zero-shot medical vision-language modeling via **Microsoft BiomedCLIP** (PubMedBERT + ViT-B/16).
- **Vascular Extraction:** Automated retinal circle detection, Graham contrast normalization, and vessel density calculation.
- **Explainability (LayerCAM):** Spatially resolved positive gradient activation maps with **anatomical optic disc suppression** to eliminate false-positive optic nerve activations.
- **Clinical Decision Support:** Real-time risk categorization, multi-model consensus alerts, and evidence-based clinical guidance.

---

## 🏥 Classification Severity Classes
- **Class 0:** No DR (Normal healthy retina)
- **Class 1:** Mild NPDR (Microaneurysms only)
- **Class 2:** Moderate NPDR (Microaneurysms, hemorrhages, hard exudates)
- **Class 3:** Severe NPDR (Extensive hemorrhages in 4 quadrants, venous beading, IRMA)
- **Class 4:** Proliferative DR (Neovascularization, vitreous/preretinal hemorrhage)

---

## 📊 Performance Benchmark
Trained and evaluated on pooled **APTOS 2019 + DDR** benchmark datasets:
- **Quadratic Weighted Kappa (QWK):** **0.9008**
- **Test Accuracy:** **84.75%**
- **Macro AUC:** **0.9707**

---

## ⚠️ Disclaimer
This system is an AI research prototype for educational and screening assistance. It does not replace professional diagnosis by a certified ophthalmologist or retina specialist.
