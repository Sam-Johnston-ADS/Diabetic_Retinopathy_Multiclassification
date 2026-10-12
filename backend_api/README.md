# 🚀 RetinaVision AI — FastAPI Inference Backend

Provides high-performance REST inference and LayerCAM explainability for the Diabetic Retinopathy screening pipeline.

---

## 🏃 Running the Backend

### 1. Run Locally
From this project directory:
```bash
# Using the virtual environment
.venv/bin/uvicorn backend_api.main:app --host 0.0.0.0 --port 8000
```
Your API will be running at: `http://localhost:8000`
Interactive Swagger Docs will be available at: `http://localhost:8000/docs`

---

### 2. Expose Publicly for Free (Cloudflare Tunnel)
If your Vercel frontend is running live on `https://your-project.vercel.app`, browsers cannot call `http://localhost:8000` from external devices. You can expose your local backend to a public HTTPS URL for free using Cloudflare Tunnel:

```bash
# Download cloudflared (Linux)
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared-linux-amd64
sudo mv cloudflared-linux-amd64 /usr/local/bin/cloudflared

# Start tunnel to port 8000
cloudflared tunnel --url http://localhost:8000
```
Cloudflare will give you a free, public HTTPS URL (e.g. `https://rapid-words-example.trycloudflare.com`).  
Paste that URL directly into the **API Endpoint** box on your Vercel web app!

---

### 3. Run on Free Google Colab (Free T4 GPU)
If you want the model to run on a free cloud GPU:
1. Open Google Colab and set Runtime to **T4 GPU**.
2. Clone your repository or upload `backend_api` and `best_model_qwk.pth`.
3. Run:
   ```python
   !pip install -r backend_api/requirements.txt
   !curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
   !dpkg -i cloudflared.deb
   
   import subprocess
   subprocess.Popen(["uvicorn", "backend_api.main:app", "--port", "8000"])
   !cloudflared tunnel --url http://localhost:8000
   ```
4. Copy the generated URL into your Vercel frontend.
