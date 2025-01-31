# LiteCNNRice
## Rice Leaf Disease Classification using Resource-Efficient Deep Learning Models via Response-Based Knowledge Distillation

###**To view the steamlit app click link https://litecnn4rice.streamlit.app/**###

---

### 🌾 **Overview** 🌾

Welcome to the **LiteCNNRice** repository, where we demonstrate an advanced approach to **rice leaf disease classification** using **resource-efficient deep learning models** via **Response-Based Knowledge Distillation (KD)**. This method enables the transfer of knowledge from a complex teacher model to a smaller student model, ensuring both high classification accuracy and a low computational footprint.

> "Empowering precision agriculture through lightweight AI models."

🔍 This repository contains:
- A **lightweight model** trained using KD.
- **Model analysis** and performance evaluation.
- The **dataset** used for training, validation, and testing.

The associated paper titled **"Rice Leaf Disease Classification using Resource-Efficient Deep Learning Models via Response-Based Knowledge Distillation"** outlines the methodology and results.

---

### 📁 **Contents** 📁

#### **1. Model Files** 🧠
- `model.h5`: The trained student model after distillation (112KB).
- `model.json`: Architecture of the model in JSON format.

#### **2. Model Analysis** 📊
- `Model_Analysis.ipynb`: Jupyter notebook that details the model’s performance evaluation and metrics, such as accuracy and confusion matrix.

#### **3. Dataset** 🌾
- `train/`: Training dataset folder with labeled images.
- `val/`: Validation dataset folder.
- `test/`: Test dataset folder.

---

### 🔑 **Key Features** ✨

#### **1. Resource Efficiency** 🌍
By utilizing **Response-Based Knowledge Distillation**, this project demonstrates how to train a smaller model while preserving or even enhancing classification performance. This approach significantly reduces computational requirements, making it suitable for deployment in resource-constrained environments like mobile devices or embedded systems.

#### **2. Classification Task** 🦠
The model is designed to classify various types of **rice leaf diseases** based on image inputs. Early disease detection can help farmers take preventative action and improve crop yields.

---

### 🚀 **Getting Started** 🚀

#### Prerequisites 🛠️

To run the code and evaluate the model, you will need the following:

- Python 3.10
- TensorFlow(version==2.15.0) / Keras
- NumPy, Pandas, Matplotlib, Seaborn (for data manipulation and visualization)
- OpenCV (for image processing)

You can install the required dependencies using the following:

```bash
pip install -r requirements.txt
