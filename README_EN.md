# Bitcoin AML Detection using Graph Neural Networks

> **Language:** [Русский](README.md) | [English](README_EN.md)

A project for detecting money laundering (AML) in Bitcoin transactions using Graph Neural Networks (GNN).

## Description

This project uses graph neural networks to analyze Bitcoin transactions and identify suspicious activity. The model is based on the Chebyshev Convolution (ChebConv) architecture and trained on the Elliptic dataset.

## Motivation

### Problem

Money laundering represents a serious threat to the financial system and is one of the primary methods for legitimizing criminal proceeds. With the emergence of cryptocurrencies, especially Bitcoin, criminals have gained new opportunities for anonymous transactions, significantly complicating the task for financial regulators and law enforcement agencies.

Traditional methods for detecting suspicious transactions, based on rules and heuristics, often fail to cope with modern money laundering schemes, which are becoming increasingly sophisticated. More advanced methods are needed that can identify complex patterns and relationships between transactions.

### Solution

Graph Neural Networks (GNN) represent a powerful tool for analyzing structured data where not only the attributes of individual objects are important, but also the connections between them. In the context of the Bitcoin blockchain, transactions naturally form a graph where nodes represent transactions or addresses, and edges represent connections between them.

Advantages of using GNN for money laundering detection:

1. **Structural relationship analysis**: GNN can analyze not only features of individual transactions but also the structure of the transaction graph, identifying complex interaction patterns
2. **Hidden pattern detection**: The model can find non-obvious connections between transactions that are not visible when analyzing individual transactions
3. **Scalability**: GNN can efficiently process large transaction graphs
4. **Adaptability**: The model can learn from real data and adapt to new money laundering schemes

### Scientific Foundation

This work is based on research in the field of applying graph neural networks for financial analysis and fraud detection. Key references:

- **"Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics"** — research on applying graph convolutional networks to analyze Bitcoin transactions and detect money laundering. The work demonstrates the effectiveness of GCN for identifying suspicious patterns in the blockchain.

- **"The Shape of Money Laundering: Subgraph Representation Learning on the Blockchain with the Elliptic2 Dataset"** — research on learning subgraph representations for detecting money laundering on the blockchain. The work uses the Elliptic2 dataset and shows how subgraph representations can improve detection of suspicious activity.

These studies show that graph neural networks outperform traditional machine learning methods in money laundering detection tasks in cryptocurrencies, achieving higher metrics for accuracy, recall, and F1-score.

### Practical Significance

The developed system can be used by:

- **Financial regulators** for automatic transaction monitoring and identification of suspicious activity
- **Cryptocurrency exchanges** for compliance and preventing the use of the platform for money laundering
- **Law enforcement agencies** for investigating financial crimes
- **Analytics companies** for providing blockchain analysis services

For more details on scientific references, see the [`research_paper/`](research_paper/) directory.

## Project Structure

```
diploma/
├── src/                    # Source code (main library)
│   ├── api/                # REST API service for Rust integration
│   │   ├── models.py      # Pydantic models for validation
│   │   ├── data_adapter.py # JSON → graph transformation
│   │   ├── response_adapter.py # Model → JSON transformation
│   │   ├── service.py      # FastAPI application
│   │   └── run_server.py  # Server startup script
│   ├── models/            # Neural network models (public)
│   │   └── custom_gat/    # Custom GAT models
│   ├── utils.py           # Utility functions
│   └── utils/             # Additional utilities
├── training/              # Model training code (confidential)
│   ├── main.py            # Main training script
│   ├── train.py           # Training and testing functions
│   └── loader.py          # Data loading and preparation for training
├── data/                  # Data (not included in repository)
│   └── elliptic/          # Elliptic dataset
├── models/                # Trained models (confidential, not in repository)
├── output/                # Training results and metrics
├── research_paper/        # Scientific references and papers
│   ├── Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics.pdf
│   └── The Shape of Money Laundering: Subgraph Representation Learning on the Blockchain with the Elliptic2 Dataset.pdf
├── config.yaml           # Model configuration
├── requirements.txt      # Python dependencies
└── README.md            # This file (Russian version)

```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd diploma
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare the data:
   - Place the Elliptic dataset in the `data/elliptic/dataset/elliptic_plus/` directory
   - Or use the `src/download_data.py` script to download the data

2. Configure:
   - Edit `config.yaml` to change model hyperparameters

3. Run training:
```bash
cd training
python main.py
```

**Note:** Training code is located in the `training/` directory and is not included in the public repository for confidentiality.

## Configuration

Main parameters in `config.yaml`:

- `data_path`: path to data
- `use_cuda`: use GPU (True/False)
- `hidden_units`: number of hidden neurons
- `hidden_units_noAgg`: number of neurons without aggregation
- `epochs`: number of training epochs
- `num_classes`: number of classes
- `lr`: learning rate
- `weight_decay`: regularization coefficient

## Model

### Architecture

The project uses the `ChebyshevConvolution` model based on ChebConv layers from PyTorch Geometric.

**Note:** Model code is located in the `training/models.py` directory and is not included in the public repository for confidentiality.

Architecture includes:
- Three convolutional layers with kernel sizes [1, 2, 4]
- ReLU6 activation functions
- Classification into 2 classes (legal/illegal transactions)

### Input Data

The model expects graph-structured data in PyTorch Geometric `Data` format:

**Input:**
- `x` (torch.Tensor): Node feature matrix of size `[num_nodes, num_features]`
  - `num_nodes`: number of nodes (transactions) in the graph
  - `num_features`: number of features for each node (depends on configuration, usually 94-166 features)
  - Type: `torch.float32`
  
- `edge_index` (torch.Tensor): Edge index matrix of size `[2, num_edges]`
  - First row: source node indices
  - Second row: target node indices
  - Type: `torch.long`
  
- `y` (torch.Tensor): Class labels of size `[num_nodes]`
  - Values: 0 (legal transaction) or 1 (illegal transaction)
  - Type: `torch.long`

**Data Format:**
- Data should be represented as a `torch_geometric.data.Data` object
- Graph should be undirected (edges represented in both directions)
- Node features should be normalized/scaled

### Output Data

**Output:**
- `out` (torch.Tensor): Classification logits of size `[num_nodes, num_classes]`
  - `num_classes`: number of classes (2 for binary classification)
  - Type: `torch.float32`
  - To get predictions: `pred = out.argmax(dim=1)` (returns class indices 0 or 1)
  
- `edge_index` (torch.Tensor): Graph structure, passed unchanged

**Usage:**
```python
out, edge_index = model((data.x, data.edge_index))
predictions = out.argmax(dim=1)  # Get predicted classes
probabilities = torch.softmax(out, dim=1)  # Get class probabilities
```

### Model Parameters

- `hidden_units`: number of neurons in hidden layers (default: 128)
- `kernel`: convolution kernel sizes for each layer `[1, 2, 4]`
- `num_classes`: number of output classes (2 for binary classification)

## Metrics

The model is evaluated using the following metrics:
- **Precision** - the proportion of correctly predicted illicit transactions among all predicted as illicit
- **Recall** - the proportion of correctly predicted illicit transactions among all actually illicit transactions
- **F1 Score** - harmonic mean between Precision and Recall
- **F1 Micro Average** - micro-averaged F1-score across all classes

Results are saved to `output/metrics_robust_base_tx_wallet.csv` and visualized as graphs.

### Metrics Visualization

#### Metrics for illicit class (illegal transactions)

![Model Metrics - Metrics Comparison](Picture%201.png)

The chart shows a comparison of Precision, Recall, F1 Score, and F1 Micro Average metrics for the ChebyshevConvolution model.

#### Aggregated Metrics by Classifier

![Aggregated Model Metrics](Picture%201_2.png)

The chart shows aggregated metrics as stacked bars, where each metric is represented by a separate segment.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Dependencies

Main dependencies:
- PyTorch
- PyTorch Geometric
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- NetworkX

See `requirements.txt` for the full list of dependencies.

## API Service

The project includes a REST API service for integration with Rust applications.

### Starting the API Service

```bash
cd src
python -m api.run_server
```

The service will be available at: `http://localhost:8000`

### Endpoints

- `POST /api/v1/analyze` - Analyze a single transaction
- `POST /api/v1/batch_analyze` - Batch transaction processing
- `GET /health` - Health check

### Data Format

The API accepts JSON requests from Rust applications in the format described in `src/api/README.md`.

**Important:** The model is not modified. All data transformations are performed through adapters:
- `data_adapter.py` - converts JSON request to graph structure
- `response_adapter.py` - converts model output to JSON response

For more details, see the documentation in `src/api/README.md`.

## Authors

Marat Kiiamov (kiya.marat@gmail.com)

## Acknowledgments

- Elliptic for providing the dataset
- PyTorch Geometric for the library for working with graph neural networks

---

**Language versions:**
- [Русский (Russian)](README.md)
- [English](README_EN.md)

