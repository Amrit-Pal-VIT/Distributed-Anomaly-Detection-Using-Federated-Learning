# Distributed Anomaly Detection Using Federated Learning

## Publication Information

**Published in:** IEEE Access, Volume 13, 2025  
**DOI:** [10.1109/ACCESS.2025.3600450](https://doi.org/10.1109/ACCESS.2025.3600450)  
**Received:** July 4, 2025 | **Accepted:** August 6, 2025 | **Published:** August 19, 2025  

**Authors:**
- Pratyush Kumar Singh
- Abishi Chowdhury (Member, IEEE)
- Amrit Pal (Member, IEEE)

**Affiliation:** School of Computer Science and Engineering, Vellore Institute of Technology, Chennai, Tamil Nadu 600127, India

---

## Overview

This research project implements **federated learning** algorithms for distributed machine learning across decentralized data sources. The research focuses on two key applications:

1. **EMNIST Handwritten Digit/Letter Classification** - Image classification on non-IID data
2. **Video Surveillance-Based Suspicious Activity Detection** - Anomaly detection using the UCF-Crime dataset

The primary goal is to train accurate machine learning models on decentralized data without centralizing raw data, thereby preserving privacy while maintaining model performance. This is the **first comprehensive study** evaluating federated learning specifically for video-based anomaly detection in surveillance systems.

## What is Federated Learning?

Federated Learning is a distributed machine learning approach where:
- Data remains on client devices (privacy-preserving)
- Clients train local models on their own data
- A central server aggregates model updates (not raw data)
- No raw data is shared between clients or with the server

### Why Federated Learning for Surveillance?

Traditional video surveillance systems face critical challenges:
- **High Communication Costs:** Transferring HD video data to cloud servers is bandwidth-intensive
- **Computational Bottlenecks:** Centralized processing creates single points of failure
- **Privacy Concerns:** Raw surveillance data must be stored and transmitted securely
- **Latency Issues:** Network delays prevent real-time threat detection
- **Security Risks:** Centralized data storage increases attack surface

Federated Learning addresses these challenges by:
- Processing video locally on surveillance devices
- Sharing only model updates (weights, gradients) with the server
- Reducing bandwidth consumption dramatically
- Enabling real-time local detection
- Maintaining privacy throughout the process
- Distributing computational load across clients

This project compares three federated learning algorithms: **FedAvg** (weighted averaging), **FedSGD** (gradient-based), and **Unweighted FedAvg** (uniform averaging).

## Project Structure

```
Pratyush/
├── README.md
├── EMNIST experiment/
│   ├── Experiment Federated Learning for Image Classification.ipynb
│   └── Experiment (Random Clients for each round) Federated Learning for Image Classification.ipynb
│
└── experiment suspicious activity detection/
    ├── experiment Federated Learning for Suspicious Activity Detection FedAvg.ipynb
    ├── Experiment Federated Learning for Suspicious Activity Detection FedSGD.ipynb
    ├── Experiment Federated Learning for Suspicious Activity Detection Unweighted FedAvg.ipynb
    ├── epochwise metrics.txt
    ├── FedAvg/
    │   ├── 5 Clients/
    │   ├── 10 Clients/
    │   ├── 15 Clients/
    │   ├── 20 Clients/
    │   ├── 25 Clients/
    │   └── 30 Clients/
    ├── FedSGD/
    │   ├── 5 Clients/
    │   ├── 10 Clients/
    │   ├── 15 Clients/
    │   ├── 20 Clients/
    │   ├── 25 Clients/
    │   └── 30 Clients/
    └── Unweighted FedAvg/
        ├── 5 Clients/
        ├── 10 Clients/
        ├── 15 Clients/
        ├── 20 Clients/
        ├── 25 Clients/
        └── 30 Clients/
```

## Experiments

### 1. EMNIST Image Classification

**Location:** `EMNIST experiment/`

**Objective:** Train a federated learning model to classify handwritten digits and letters using the EMNIST dataset.

**Model Architecture:**
```
InputLayer (784 dimensions)
    ↓
Dense(10) with Softmax activation
    ↓
Output: 10 classes for digit classification
```

**Experiments:**
1. **Experiment Federated Learning for Image Classification.ipynb** - Fixed client set across all rounds
2. **Experiment (Random Clients for each round) Federated Learning for Image Classification.ipynb** - Random client selection each round

**Key Metrics:**
- Per-round accuracy and loss tracking
- Convergence analysis over 10 epochs

**Sample Results:**
- Epoch 1: ~59% accuracy
- Epoch 10: ~85.7% accuracy
- Validation accuracies: 9.6%-64.6% range

---

### 2. Suspicious Activity Detection (Video Surveillance)

**Location:** `experiment suspicious activity detection/`

**Objective:** Detect suspicious activities and anomalies in real-world video surveillance footage using federated learning to maintain privacy while enabling real-time threat detection.

**Dataset Details:**
- **Dataset:** UCF-Crime - Real surveillance video footage
- **Scenarios:** Realistic surveillance scenarios from public cameras
- **Input Format:** Extracted video frames and features
- **Distribution:** Non-IID (heterogeneous activity patterns across locations)
- **Real-World Focus:** This is the first FL study specifically on video-based anomaly detection

**Model Architecture:**
```
InputLayer (784 dimensions)
    ↓
Dense(256) + BatchNormalization + Dropout(0.3)
    ↓
Dense(128) + BatchNormalization + Dropout(0.4)
    ↓
Dense(64) + BatchNormalization
    ↓
Dense(32)
    ↓
Dense(16)
    ↓
Dense(14) with Softmax activation
    ↓
Output: 14-class classification
```

**Regularization:** L2(0.001) penalty to prevent overfitting

**Federated Learning Algorithms Tested:**

| Algorithm | Subdirectory | Aggregation Strategy | Description | Best Accuracy |
|-----------|--------------|------------------|-------------|----------------|
| **FedAvg** | `FedAvg/` | Weighted by client data size | Standard federated averaging - weights updates by number of local samples | 82% (30 clients) |
| **FedSGD** | `FedSGD/` | Gradient-based (no averaging) | Pure stochastic gradient descent aggregation at the server | 41% (max) |
| **Unweighted FedAvg** | `Unweighted FedAvg/` | Uniform averaging (RECOMMENDED) | Equal contribution from each client, optimal for non-IID data | **85% (25 clients)** ✓ |

**Client Configurations:** Each algorithm tested with 5, 10, 15, 20, 25, and 30 clients.

**Training Configuration:**
- **Optimizer:** SGDM (Stochastic Gradient Descent with Momentum)
- **Client Learning Rate:** 0.02
- **Server Learning Rate:** 1.0
- **Rounds:** 50 federated averaging rounds per configuration
- **Local Epochs (E):** Specified for each client
- **Batch Size (B):** Mini-batch size for local training
- **Aggregation:** Server-side model update aggregation

**Key Findings from Experiments:**

| Algorithm | 5 Clients | 10 Clients | 15 Clients | 20 Clients | 25 Clients | 30 Clients |
|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| **FedAvg Accuracy** | 0.73 | 0.75 | 0.77 | 0.79 | 0.80 | **0.82** |
| **FedAvg Loss** | 0.81 | 0.77 | 0.73 | 0.70 | 0.67 | 0.64 |
| **Unweighted FedAvg Accuracy** | 0.74 | 0.78 | 0.81 | 0.82 | **0.85** ✓ | 0.84 |
| **Unweighted FedAvg Loss** | 0.78 | 0.72 | 0.64 | 0.59 | **0.53** ✓ | 0.56 |
| **FedSGD Accuracy** | - | - | - | - | - | 0.41 (max) |
| **FedSGD Loss** | - | - | - | - | - | 1.59-1.91 |

**Results Structure:**

Each experiment configuration directory contains:
- **Roundwise Metrics [N] Clients.txt** - Accuracy and loss tracked per round
- **Roundwise Updates [N] Clients.txt** - Global model weight changes per round

**Notebooks:**
1. **experiment Federated Learning for Suspicious Activity Detection FedAvg.ipynb** - FedAvg algorithm implementation
2. **Experiment Federated Learning for Suspicious Activity Detection FedSGD.ipynb** - FedSGD algorithm implementation
3. **Experiment Federated Learning for Suspicious Activity Detection Unweighted FedAvg.ipynb** - Unweighted FedAvg implementation (RECOMMENDED)

---

## Key Technologies & Libraries

- **TensorFlow Federated (TFF):** Primary federated learning framework
- **TensorFlow/Keras:** Model definition and training
- **Python:** Core programming language
- **Jupyter Notebooks:** Experiment execution and documentation
- **Google Colab:** Runtime environment (with Drive integration)
- **scikit-learn:** Advanced metrics (Precision, Recall, F1-Score)
- **NumPy/Pandas:** Data manipulation
- **Matplotlib:** Results visualization

## Main Features

**Privacy-Preserving:** Raw client data never leaves client devices  
**Non-IID Data:** Experiments with realistic heterogeneous data distributions  
**Multiple Algorithms:** Comparative analysis of FedAvg, FedSGD, and variants  
**Scalability Testing:** 5-30 client scenarios to evaluate algorithm performance  
**Comprehensive Metrics:** Accuracy, loss, precision, recall, and F1-score tracking  
**Two Domains:** Computer vision (EMNIST) and network security (anomaly detection)  

## Data Characteristics

### EMNIST - Non-IID by Handwriter
- Each client owns data from a single handwriter
- Unique writing styles create natural data heterogeneity
- Simulates real federated scenarios where clients have distinct data distributions

### UCF-Crime - Non-IID Video Surveillance Data
- Clients represent different surveillance camera locations
- Activity patterns vary significantly across locations (natural heterogeneity)
- Different cameras capture diverse suspicious activities and normal behaviors
- Tests algorithm robustness to realistic distributed surveillance data
- **Non-IID is intentional:** Reflects real-world surveillance system deployment

## Key Findings & Recommendations

### Optimal Federated Learning Strategy: **Unweighted FedAvg**

Based on comprehensive experiments on the UCF-Crime dataset:

**Unweighted FedAvg achieves best performance:**
- Highest accuracy: **85%** at 25 clients
- Lowest loss: **0.53** at 25 clients  
- Consistent improvement across all client configurations
- Superior handling of non-IID data distributions

**FedAvg (weighted averaging) shows solid performance:**
- Accuracy: **82%** at 30 clients
- Loss: **0.64** at 30 clients
- Consistent convergence with increasing clients
- Acceptable for balanced data scenarios

**FedSGD is not recommended:**
- Highly erratic performance
- Maximum accuracy: **41%** (unsatisfactory)
- High loss: **1.59-1.91** range
- Poor convergence on video anomaly detection

### Why Unweighted FedAvg Works Best

1. **Non-IID Data Handling:** Equal weighting compensates for data heterogeneity across surveillance locations
2. **Prevents Dominance:** Prevents larger-data clients from dominating model aggregation
3. **Robustness:** More stable convergence across different client counts (5-25 clients)
4. **Practical Implication:** No need to track or weight by client data size

### Performance Characteristics

| Metric | FedAvg | Unweighted FedAvg | FedSGD |
|--------|--------|------------------|--------|
| **Best Accuracy** | 82% | 85% ✓ | 41% |
| **Convergence Pattern** | Steady | Consistent | Erratic |
| **Non-IID Handling** | Moderate | Excellent ✓ | Poor |
| **Scalability (5-30 clients)** | Good | Excellent ✓ | Unstable |
| **Recommended** | Alternative | Yes ✓ | No |

---

## Research Contributions

This work makes several important contributions to federated learning research:

1. **First Comprehensive FL Evaluation** for video-based anomaly detection in surveillance systems
2. **Comparative Analysis** of three aggregation strategies (FedAvg, Unweighted FedAvg, FedSGD) on realistic UCF-Crime dataset
3. **Optimal Strategy Identification** - Unweighted FedAvg recommended for non-IID surveillance data
4. **Communication Efficiency Analysis** - Only model updates transmitted, not video data
5. **Privacy-Preserving Approach** - Raw surveillance footage never leaves local cameras
6. **Practical Validation** - Non-IID data distribution reflects real surveillance deployments

---

## Advantages of Federated Learning for Surveillance

| Advantage | Benefit | Impact |
|-----------|---------|--------|
| **Privacy Preservation** | Raw video data never leaves camera devices | Complies with privacy regulations (GDPR, etc.) |
| **Reduced Bandwidth** | Only model updates transmitted, not video streams | Dramatic cost reduction in surveillance infrastructure |
| **Computational Efficiency** | Load distributed across cameras | Eliminates centralized server bottleneck |
| **Latency Reduction** | Real-time local anomaly detection | Faster threat response |
| **Scalability** | System scales with camera participation | Viable for large-scale city-wide deployments |
| **Security Enhancement** | No centralized data vulnerability | Reduced attack surface and data breach risk |
| **Continuous Operation** | Local processing survives network outages | Cameras function independently if connection lost |

---

## Research Questions Addressed

1. **Can federated learning maintain accuracy for video anomaly detection?**
   - Yes: Unweighted FedAvg achieves 85% accuracy without centralizing video data

2. **How does client weighting affect non-IID data handling?**
   - Unweighted averaging outperforms weighted FedAvg for heterogeneous data

3. **How many clients optimize performance?**
   - Sweet spot: 25 clients for Unweighted FedAvg, 30 clients for FedAvg

4. **Is federated learning applicable across domains?**
   - Yes: Successfully demonstrated on EMNIST (vision) and UCF-Crime (surveillance)

## Implementation & Execution

### How to Run Experiments

1. **Open Jupyter Notebooks** in Google Colab or local Jupyter environment
2. **Configure Data Sources:** Point to EMNIST dataset or UCF-Crime TFRecords
3. **Run Experiments:** Execute notebook cells to start federated learning training
4. **Monitor Results:** Track per-round metrics and model convergence
5. **Compare Algorithms:** Analyze outputs from different federated learning approaches

### Results Storage

- **Roundwise Metrics:** Plain text files with accuracy/loss per round
- **Model Updates:** Layer-by-layer weight changes logged per round
- **Visualizations:** Matplotlib plots showing convergence curves

## Technical Details

### Federated Learning Formulas

**FedAvg (Weighted Aggregation):**
$$w_t = \sum_{i=1}^{K} \frac{n_i}{n} w_{i,t}$$

where $n_i$ is the number of samples on client $i$, $n$ is total samples, and $w_{i,t}$ is client weights at round $t$.

**Unweighted FedAvg:**
$$w_t = \frac{1}{K} \sum_{i=1}^{K} w_{i,t}$$

where each client contributes equally regardless of data size.

**FedSGD (Baseline):**
- Clients send individual gradients
- Server performs SGD updates
- Different convergence characteristics

### Key Hyperparameters

- **K** (Number of clients): 5, 10, 15, 20, 25, 30
- **E** (Local epochs): As specified per configuration
- **B** (Batch size): Mini-batch size for local training
- **η** (Learning rate - client): 0.02
- **η** (Learning rate - server): 1.0
- **Rounds:** 50 federated averaging rounds
- **Optimizer:** SGDM (Stochastic Gradient Descent with Momentum)

---

## Important Notes

- All experiments implemented in Jupyter Notebooks for reproducibility
- Models compatible with TensorFlow Federated (TFF) framework
- Non-IID data distribution intentional for realistic evaluation
- Results files organized by client count and algorithm configuration
- Experiments portable between Google Colab and local Jupyter environments
- No raw data (video or images) ever centralized
- Communication overhead analysis included in paper

### Reference to Paper

For detailed methodology, results analysis, discussion, and conclusions, refer to:

**Distributed Anomaly Detection Using Federated Learning**  
- Published in: IEEE Access, Volume 13, 2025
- DOI: [10.1109/ACCESS.2025.3600450](https://doi.org/10.1109/ACCESS.2025.3600450)
- Available: Open Access

---

**Project Focus:** Demonstrating practical federated learning implementation for privacy-preserving distributed video anomaly detection, establishing best practices for surveillance system deployment with emphasis on communication efficiency and privacy preservation.
