# ICI_response_project_2025
Immune checkpoint inhibitor therapy Response prediction #Bioinformatics #ComputerScience #Incheon National University (South Korea) 

# Overview
--- 
This project adapts and extends BioXNet (a biologically inspired neural network) for deciphering anticancer drug responses in the context of Immune Checkpoint Inhibitor (ICI) cohorts. The model leverages hierarchical biological structures (e.g., Reactome pathways) and multi-omics data (mutations, CNV, and gene expression) to predict patient response to immunotherapy, advancing precision med

BioXNet paper : https://www.biorxiv.org/content/10.1101/2024.01.29.576766v1

# Methods
--- 
1. Hierarchical Pathway Network :
Integrates Reactome’s pathway relationships, creating sparse mask matrices for gene-pathway and pathway-pathway connections.
Each model layer corresponds to biological entities (genes or pathways) progressively reducing dimensions from genes to high-order pathways.

2. Multi-task Learning :
Predicts patient response and classifies data into multiple domain cohorts (e.g., TCGA, Liu, Hugo).
Includes gradient reversal layers for adversarial domain adaptation.

3. Attention Mechanisms :
Assigns attention weights to highlight biologically important features contributing to predictions.

4. Weight Transfer :
Fine-tunes the pre-trained model (from TCGA data) for predicting drug responses in other ICI cohorts (e.g., Liu, Hugo).

5. Data Integration :
Combines genomic profiles (methylation, mutation, CNV) and clinical data to enhance interpretability and performance.

# Model Architecture:
---
+ Shared Encoder: Hierarchical structure from genes to pathways.(BioXNet)
+ Response Classifier: Predicts drug response.
+ Domain Classifier: Adapts to cohort variations using adversarial learning.

# Input Data
---
+ Genomic Data: CNV, mutation, gene expression for 6,640 genes derived from TCGA and other cohorts.
+ Pathway Relationships: Reactome’s pathway hierarchy. Gene-to-pathway associations.
+ Clinical Features: Cancer type, treatment type, gender, age.






