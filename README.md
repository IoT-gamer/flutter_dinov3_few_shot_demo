## DINOv3 Few-Shot Segmentation Demo

This repository contains a cross-platform solution for **Few-Shot Instance Segmentation** on mobile devices. It leverages a split-model architecture using **DINOv3** as a static feature extractor and a lightweight, dynamically trained **Logistic Regression** classifier for specific objects.

## 🏗️ Architecture Overview
The project is split into a Flutter mobile client and a Python backend to handle training.

**1. Mobile Client (Flutter)**

* **Split-Inference Pipeline:** The app runs a heavy DINOv3 model once to extract features and then "hot-swaps" tiny `.onnx` classifiers to identify different objects in real-time.

* **State Management:** Uses **Bloc/Cubit** to manage camera streams, model loading states, and UI overlays.

* **High Performance:** Heavy computation is offloaded to a Background Isolate using `integral_isolates` and `flutter_onnxruntime`.

**2. Backend (Python & PocketBase)**

* **PocketBase:** Acts as the central hub for storing training images (datasets) and serving trained `.onnx` models.

* **Dagster Pipeline:** An automated orchestration engine that senses new uploads, extracts features via DINOv3, trains a Logistic Regression model, and exports it back to the client.

## 🚀 Getting Started

### **Prerequisites**

* **Flutter SDK** (Latest stable)
* **Python 3.10+**
* **PocketBase** (Running instance)
* **DINOv3 ONNX Model:** You must export the DINOv3 feature extractor and place it in the `assets/` folder.
    * [Export Notebook for DINOv3 ONNX](https://github.com/IoT-gamer/segment-anything-dinov3-onnx/blob/main/notebooks/dinov3_onnx_export.ipynb)

### **Installation**

**1. Backend Setup**
```bash
cd python
pip install -r requirements.txt
```

Update the `POCKETBASE_URL` and admin credentials in `training_pipeline.py`. Launch the Dagster webserver:
```bash
dagster dev -f training_pipeline.py
```
**2. Flutter Setup**
1. Add the `dinov3_feature_extractor.onnx` to `assets/`.

2. Update `kPocketBaseUrl` in `lib/services/pocketbase_service.dart` to match your local IP.

3. Install dependencies and run:
```bash
flutter pub get
flutter run
```

## 📱 Workflow
**1. Capture Data:** Use the **Model Manager** screen to upload photos of an object. Use images with transparent backgrounds (created via the [Object Mask App](https://github.com/IoT-gamer/flutter_segment_anything_app)).

**2. Remote Training:** The Python backend detects the new dataset via Dagster sensors, trains a classifier, and marks it as "ready".

**3. Deploy:** Click **Download & Use** in the app to hot-swap the new model into the live camera feed.

**3. Segment:** Toggle the "Play" button on the camera screen to see the real-time segmentation overlay.

## 🛠️ Configuration & Tools
| Feature | Component |
| :--- | :---: |
**Inference Engine**	| ONNX Runtime (with NNAPI/CoreML support) |
**Image Processing**	| OpenCV (Dart/C++) |
**Orchestration**	| Dagster |
**Backend/DB**	| PocketBase |

## 🙏 Acknowledgements
- This work builds upon the official implementations and research from the following projects:

    **DINOv3:** [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)