# Retrosynthesis Pathway Optimization Augmented by Molecular Analogs

This repository implements an end-to-end machine learning pipeline that proposes and scores synthetic pathways for known drug molecules. Our system combines a reaction yield predictor based on graph neural networks (GNNs) with a SMILES-based VAE for molecular analog generation. Together, these components enable speculative retrosynthetic planning by suggesting high-yield reaction pathways, expanding the synthesis search space through analog substitution. Users can run the pipeline in the retrosynthesis_model_pipeline.ipynb notebook.

Developed as the final project for CHEM 277B – Spring 2025, UC Berkeley MSSE.

Authors: Haris Saeed, Paul Graggs, Girnar Joshi, Festo Muhire

---

## Motivation

- Drug discovery is inherently complex, costly, and time-consuming, with high failure rates. Despite numerous emerging technologies, finding efficient synthesis routes remains difficult.
- The drug discovery field is highly competitive with many companies investing heavily, which increases pressure to innovate and reduce costs while accelerating timelines.
- Using AI to explore alternative synthesis pathways can enhance the availability of known drugs and lower costs by predicting and optimizing high-yield production routes efficiently.

---

## Components

### 1. The Pipeline
Given a target SMILES, this module:
- Splits it into synthons using RDKit BRICS
- Searches for matching or similar reactants in the dataset
- Uses a trained GNN to predict yield of forming the target from those reactants

### 2. The System
Builds multi-step retrosynthetic pathways by recursively applying the pipeline to intermediate reactants.

### 3. The Process
Runs the system multiple times to generate multiple unique pathways, scoring each and selecting the highest-yield route.

---

## ML Models

### Yield Predictor (GNN)
- Trained on USPTO-Applications dataset (~2M reactions)
- Uses PyTorch Geometric graph objects (atom and bond features)
- Architecture: Graph Attention Network (GAT) → MLP predictor
- Goal: Predict reaction yield (0–100%) for reactant → product conversions

### Molecular Analog Generator (SMILES-VAE)
- Trained on synthons generated via BRICS from 10k SMILES
- GRU-based VAE trained with cross-entropy + KL divergence loss
- Analog generation via latent noise + top-k sampling
- Only valid SMILES are returned via RDKit filtering


