# TriplePhase-Healthcare-AI-Privacy
Official implementation of "Triple-Phase Privacy Shielding for Healthcare AI: A Federated and Blockchain-Enabled Framework with Formal Privacy Guarantees". This repository contains the source code, experiments, and analysis for our privacy-preserving healthcare AI framework combining federated learning with blockchain technology.
## Overview
This repository contains the complete implementation of our triple-phase privacy framework for healthcare AI, combining federated learning with blockchain technology to provide formal privacy guarantees.
## Repository Structure
TriplePhase-Healthcare-AI-Privacy/Data
├── figures
│ ├── 3d_overhead.pdf
│ ├── risk_matrix_assessment.png
│ └── smooth_tradeoff.pdf
├── Hospital --- 01
│ ├── Step - 02 --- TPPS
│ │ ├── softlabels_anonymized.csv
│ │ ├── softlabels_encrypted.enc
│ │ ├── softlabels_encryption_key.key
│ │ ├── softlabels_noised.csv
│ │ ├── softlabels_processing_readme.json
│ │ ├── softlabels_synthetic.csv
│ │ └── TPPS_PHASES.py
│ ├── Step - 03 --- TPPS-KG
│ │ ├── hospital_1_LKG.pkl
│ │ ├── hospital_1_LKG.rdf
│ │ ├── hospital_1_LKG.ttl
│ │ ├── hospital_1_LKG_embeddings.json
│ │ ├── hospital_1_LKG_preprocessed.csv
│ │ ├── Local_KG.py
│ │ └── softlabels_noised.csv
│ └── Steps - 01 --- Creating Softlabels and FKD
│ ├── cleveland_raw.csv
│ ├── cleveland_softlabels.csv
│ ├── combined_phase_fkd.csv
│ ├── hungarian_raw.csv
│ ├── hungarian_softlabels.csv
│ └── step-1-processed.py
├── Hospital --- 02
│ ├── Step - 02 --- TPPS
│ │ ├── softlabels_anonymized.csv
│ │ ├── softlabels_encrypted.enc
│ │ ├── softlabels_encryption_key.key
│ │ ├── softlabels_noised.csv
│ │ ├── softlabels_processing_readme.json
│ │ ├── softlabels_synthetic.csv
│ │ └── TPPS_PHASES.py
│ ├── Step - 03 --- TPPS-KG
│ │ ├── hospital_2_LKG.pkl
│ │ ├── hospital_2_LKG.rdf
│ │ ├── hospital_2_LKG.ttl
│ │ └── softlabels_noised.csv
│ └── Steps - 01 --- Creating Softlabels and FKD
│ ├── cleveland_raw.csv
│ ├── cleveland_softlabels.csv
│ ├── combined_phase_fkd.csv
│ ├── hungarian_raw.csv
│ ├── hungarian_softlabels.csv
│ └── step-1-processed.py
└── Validation LKG
├── Hospital -- 01
│ ├── Functional_Validation.py
│ ├── hospital_1_LKG.ttl
│ ├── Structural_Validation.py
│ └── Technical_Validation.py
└── Hospital -- 02
├── Functional_Validation.py
├── hospital_2_LKG.ttl
├── Structural_Validation.py
└── Technical_Validation.py
