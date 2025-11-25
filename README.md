Official implementation of **"A Context-Aware Transformer for Corporate Default Prediction: Integrating Firm-Level Financials with Industry and Macroeconomic Factors"** - a novel deep learning framework that holistically integrates multi-level contextual signals for corporate default prediction.

## 📖 Abstract

Corporate default prediction using firm-level financial panel data is fundamental to modern credit risk management. While existing approaches have demonstrated predictive value, they largely overlook two critical contextual factors: industry-specific dynamics and macroeconomic conditions. To address this limitation, we propose an end-to-end Transformer-based framework designed to holistically integrate firm-level temporal patterns with industry and macroeconomic contexts. Our model achieves state-of-the-art results with an overall accuracy of **83.98%**, representing a **3.06%** improvement over the best-performing baseline, along with a **5.84%** enhancement in macro-F1 score.

## 🚀 Key Features

- **Multi-Level Context Integration**: Seamlessly combines firm-level financials, industry categorizations, and macroeconomic indicators
- **Structured Attention Mechanism**: Enforces temporal causality while enabling rich cross-context interactions
- **Comprehensive Dataset**: Publicly available multi-source dataset for Chinese A-share listed companies
- **State-of-the-Art Performance**: Outperforms 11 baseline models across multiple evaluation metrics
- **Interpretability**: Provides insights through attention visualization and SHAP analysis

## 📊 Dataset

The dataset integrates three primary data components:

- **Firm-Level Financial Data**: 78 financial features from CSMAR database (2014-2024)
- **Macroeconomic Data**: 36 indicators from AKShare library
- **Credit Rating Data**: Five-class ratings from Wind database

**Total**: 14,789 firm-year observations across 5 industries

### Dataset Access
The complete dataset is available at: [Google Drive Link](https://drive.google.com/drive/folders/1p314NhGGAX5tttvclkY6E7CmedQh1CAo?usp=sharing)
