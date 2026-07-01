# 🛰️ Route Resilience
**AI-Powered Occlusion Robust Road Extraction and Graph-Theoretic Urban Road Intelligence**

> Built for the **ISRO Bharatiya Antariksh Hackathon**

---

## 🎯 Overview

Route Resilience extracts road networks from satellite imagery under challenging occlusion conditions (trees, shadows, clouds, vehicles), reconstructs topologically correct road graphs, and performs structural vulnerability analysis through disaster simulation.

---

## 🚀 Features

| Feature | Description |
|---|---|
| 🛣️ Road Segmentation | DeepLabV3+ / U-Net with occlusion robustness |
| 🕸️ Graph Reconstruction | Skeleton → Graph → MST Healing |
| ⚠️ Bottleneck Detection | Betweenness Centrality + Gatekeeper Nodes |
| 💥 Disaster Simulation | Node/Edge failure → Resilience Index |
| 📊 Interactive Dashboard | 7-page Streamlit app with Folium maps |
| 🗂️ Dataset Support | SpaceNet, DeepGlobe, OpenSatMap, OSM |

---

## 📁 Project Structure

```
Route-Resilience/
├── analysis/           # Centrality analysis & disaster simulation
├── config/             # Pydantic configuration system
├── core/               # Interfaces & data models
├── dashboard/          # Streamlit dashboard (app.py)
├── datasets/           # Dataset loaders & factory
├── graph/              # Skeleton extraction, graph building, healing
├── preprocessing/      # Image preprocessing & geospatial handling
├── segmentation/       # DeepLabV3+ & U-Net wrappers
├── utils/              # Error classes & utilities
├── visualization/      # Matplotlib graph rendering
└── requirements.txt
```

---

## ⚡ Quick Start

```bash
# 1. Clone
git clone https://github.com/jayjit-2025/Route-Resilience.git
cd Route-Resilience

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run dashboard
python -m streamlit run dashboard/app.py
```

Open **http://localhost:8501** in your browser.

---

## 📊 Demo Workflow

```
Upload Satellite Image
        ↓
Road Detection (DeepLabV3+)
        ↓
Skeleton Extraction (scikit-image)
        ↓
Road Graph Construction (NetworkX)
        ↓
MST Graph Healing
        ↓
Critical Bottleneck Detection
        ↓
Disaster Simulation
        ↓
Interactive Dashboard
```

---

## 🗂️ Dataset Setup

See the **Dataset Info** page in the dashboard for setup instructions, or download the guide from within the app.

Supported datasets:
- **SpaceNet Roads** — https://spacenet.ai/roads/
- **DeepGlobe Road Extraction** — https://competitions.codalab.org/competitions/18467
- **OpenSatMap** — https://opensatmap.github.io/
- **OSM** (auto-fetched via OSMnx)

---

## 📐 Evaluation Metrics

| Metric | Description |
|---|---|
| IoU | Intersection over Union |
| Dice Score | F1-like overlap metric |
| Relaxed IoU | 3-5 pixel tolerance buffer |
| Connectivity Ratio | Largest component / total nodes |
| Topological Accuracy | Average Path Length error vs OSM |
| Resilience Index | Composite robustness score [0,1] |

---

## 🔧 Tech Stack

Python · PyTorch · NetworkX · Rasterio · OpenCV · scikit-image · Streamlit · Folium · Matplotlib · Pydantic

---

## 👨‍💻 Team

Built for ISRO Bharatiya Antariksh Hackathon 2025
