# 👁️ RetinaVision AI — Vercel Web Deployment

A modern, responsive, clinical-grade web application for **5-Class Diabetic Retinopathy Screening & Explainability** deployed on **Vercel**, connecting to a high-performance **FastAPI Deep Learning Backend**.

---

## 🚀 How to Deploy on Vercel

### Option A: 1-Click Deploy via GitHub (Recommended)
1. Push this folder (`vercel_frontend`) to a new GitHub repository:
   ```bash
   cd vercel_frontend
   git init
   git add .
   git commit -m "Initial Vercel frontend commit"
   git branch -M main
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>.git
   git push -u origin main
   ```
2. Go to [vercel.com](https://vercel.com) and log in.
3. Click **Add New...** -> **Project**.
4. Import your GitHub repository.
5. Keep the Framework Preset as **Other** (Root directory has `index.html` and `vercel.json`).
6. Click **Deploy**.
7. In ~30 seconds, your site is live at: `https://<your-project>.vercel.app`!

---

### Option B: Deploy via Vercel CLI
If you have Node.js and the Vercel CLI installed:
```bash
npm i -g vercel
cd vercel_frontend
vercel
```
Follow the interactive prompts and choose default settings.

---

## 🔌 Connecting to Your FastAPI Model Backend

Once your Vercel site is live:
1. Start your FastAPI backend (see `../backend_api/README.md`).
2. Open your Vercel web app in the browser.
3. In the top bar, set the **API Endpoint** input field to your backend's URL:
   - For local testing: `http://localhost:8000`
   - For cloud/public tunnel: `https://<your-cloudflare-or-colab-tunnel-url>`
4. Click the refresh button to verify the green health badge.
5. Click any sample case or upload a fundus photograph to run the analysis!
