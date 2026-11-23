# Local Land-Use Classification Web Interface (Smart Agriculture — Milestone 3)

A lightweight web application that runs **locally on your machine** (no cloud uploads required). Upload a satellite image (EuroSAT-style), and the trained model from the notebook will predict both the detailed land-use class (10 classes) and the grouped booklet category (`Agriculture` / `Water` / `Urban` / `Roads` / `Trees`) matching **Part F** of the notebook.

## 1) Run the Notebook Through the Final Cell

Execute the entire `Smart_Agriculture_M3_Optimization.ipynb` notebook (on Google Colab with a GPU, or locally if you have a GPU/sufficient time). The final cells in the notebook (under the *"Deliverables & summary"* section) export the following two files:

```text
outputs/models/eurosat_resnet_final.pt
outputs/models/model_config.json
```

## 2) Copy the Model Files to This Project

Place the two exported files into the `model/` directory inside this project so the folder structure looks like this:

```text
smart-agri-webapp/
  model/
    eurosat_resnet_final.pt
    model_config.json
  app.py
  model_def.py
  ...
```

## 3) Install Dependencies (One-Time Setup)

```bash
cd smart-agri-webapp
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Note:** CPU-only `torch`/`torchvision` is completely sufficient for single-image inference and runs very fast locally, so there is no need to configure CUDA on your machine.

## 4) Start the Local Server

```bash
python app.py
```

Then open your browser and navigate to:

```text
http://127.0.0.1:5000
```

Upload an image and click **"Classify Image"** (*"تصنيف الصورة"*). Everything—image upload, preprocessing, inference, and result display—happens strictly on your device via `localhost`, with no internet connection or cloud services involved.

## How It Works

- **`app.py`**: A lightweight Flask server handling the upload page (`/`) and the prediction endpoint (`/predict`).
- **`model_def.py`**: Replicates the exact model architecture (`build_model` / `make_head`) and image preprocessing pipeline (`preprocess_image`) from the notebook to ensure saved weights load properly and produce identical results. It also contains the mapping from the 10 EuroSAT classes to the 5 grouped categories (`Agriculture`, `Water`, `Urban`, `Roads`, `Trees`) per Part F.
- **`templates/index.html` + `static/style.css`**: A clean drag-and-drop upload interface displaying the detailed class, grouped category, confidence score, and the top-2 alternative predictions.

## Troubleshooting: "Model files not found" Error

If you see a message indicating that the model files were not found, it means `model/eurosat_resnet_final.pt` or `model/model_config.json` is missing from the `model/` folder. Please revisit **Steps 1 and 2**.

## Note on Project Scope

The provided notebook (EuroSAT — satellite image classification into 10 land-use classes grouped into `Agriculture`/`Water`/`Urban`/`Roads`/`Trees`) differs from the *"Crop Production in India"* tabular dataset topic described in `Project Proposal.pdf`. This web interface is built to match the notebook implementation (satellite image $\rightarrow$ land-use class), so please verify that this aligns with your Milestone 3 requirements prior to final submission.
