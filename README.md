Time Series Forecasting with Various Models on ETT-h1 Dataset

This repository contains implementations of various machine learning models for time series forecasting using the ETT-h1 dataset. The models include WaveNet, Transformer, TCN, N-BEATS, LSTM, FEDformer, GRU, FFNN, CatBoost, Autoformer, Informer, and DeepAR.

Table of Contents:
    Introduction
    Dataset
    Installation
    Usage
    Preprocessing
    Running Models
    Results

Introduction:
This project aims to demonstrate the application of various state-of-the-art time series forecasting models on the ETT-h1 dataset. The dataset consists of hourly measurements of different load types and the oil temperature of an electricity transformer, which we aim to predict.

Dataset:
The ETT-h1 dataset can be downloaded from this link [https://github.com/zhouhaoyi/ETDataset]. It includes the following columns:

date: Timestamp of the recorded data.

HUFL: High-usage frequency load.

HULL: High-usage low load.

MUFL: Medium-usage frequency load.

MULL: Medium-usage low load.

LUFL: Low-usage frequency load.

LULL: Low-usage low load.

OT: Oil temperature (target variable).
