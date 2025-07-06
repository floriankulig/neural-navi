# Evaluation Framework Overview

## 🎯 Scientific Evaluation Methodology

The evaluation framework implements a comprehensive scientific approach for analyzing multimodal driving behavior prediction models. Based on empirical findings from 12 highway recordings with extreme class imbalance (1:36 for brake events), the evaluation prioritizes **PR-AUC over accuracy-based metrics** due to the realistic sparsity of safety-critical events.

### Key Experimental Findings

- **Task Selection**: Focus on 1s and 2s prediction horizons due to exponential performance degradation at longer horizons (88% drop from 1s to 5s for brake events)
- **Architecture Performance**: Transformer decoder outperforms LSTM with better stability and generalization
- **Attention Patterns**: Consistent recency bias (0.36-0.44 attention weight on last 5 positions)
- **Edge Computing**: YOLO dominates pipeline latency (>92%), multimodal model contributes <6%

---

## 🔬 Core Evaluation Scripts

### 1. **Single Model Evaluation** (`eval_single.py`)
**Purpose**: Comprehensive scientific evaluation of individual multimodal architectures

**Key Features**:
- Automatic task extraction from model checkpoints
- Optimal threshold finding for F1, MCC, and Balanced Accuracy
- Comprehensive metrics: PR-AUC, ROC-AUC, Brier Score, calibration analysis
- Scientific visualizations with confusion matrices
- Detection capability analysis vs random baseline

**Usage**:
```bash
python evaluation/eval_single.py --model data/models/multimodal/simple_concat_transformer/best_model.pt
```

**Output**: 
- Detailed metrics JSON with statistical significance tests
- Calibration analysis for prediction reliability
- Performance categorization (No/Weak/Moderate/Strong Detection Capability)

### 2. **Inference Performance Benchmark** (`inference_benchmark.py`)
**Purpose**: Scientific comparison of LSTM vs Transformer decoder inference performance

**Key Features**:
- Single-sequence inference focus (batch_size=1) for real-time scenarios
- Multiple optimization strategies: Baseline, PyTorch Compile, Mixed Precision
- Memory profiling for GPU/MPS/CPU
- Statistical analysis with confidence intervals

**Key Findings**:
- Transformer: 9.28±0.43ms (baseline) → 7.17±0.69ms (mixed precision)
- LSTM: 6.51±1.07ms (baseline) with lower memory footprint
- Mixed precision provides 29% speedup for Transformer, degrades LSTM performance

### 3. **Attention Pattern Analysis** (`attention_analyzer.py`)
**Purpose**: Deep analysis of Transformer attention mechanisms for interpretability

**Key Features**:
- Layer-wise attention weight extraction using hooks
- Temporal bias analysis (recency vs early bias)
- Task-specific attention differences
- Entropy analysis for attention concentration

**Key Findings** (from paper):
- All 4 Transformer layers develop strong recency bias (0.36-0.44 for last 5 positions vs 0.14-0.18 for first 5)
- Convergence on position 19 as primary attention peak
- Validates hypothesis that local temporal context is most relevant for brake prediction

### 4. **False Negative Intensity Analysis** (`analyze_false_negatives.py`)
**Purpose**: Analyzes intensity characteristics of missed predictions using pre-computed event intensities

**Key Features**:
- Uses HDF5 pre-computed intensity data (brake force, accelerator position)
- Statistical comparison of TP vs FN intensities
- Mann-Whitney U tests for significance
- Cohen's d effect size calculation

**Scientific Value**: Identifies if model misses weak vs strong events, critical for safety assessment

---

## 📊 Supporting Analysis Tools

### YOLO Performance Analysis

**`performance.py`**: Comprehensive YOLO model benchmark with scientific metrics
- Parameter count vs inference speed analysis
- Memory usage profiling
- Multi-metric radar charts for model comparison
- Production-ready performance expectations

**`yolo_optimization_benchmark.py`**: Systematic optimization evaluation
- Class filtering (vehicle-only detection)
- Verbose mode impact
- Optimization effectiveness quantification

### Data Analysis & Visualization

**`sequence_viewer.py`**: Interactive multimodal sequence explorer
- Synchronized telemetry + detection + image display
- Real-time sequence playback
- Export capabilities for scientific presentations

**`compression.py`**: Image compression impact on detection
- Quality vs performance trade-offs
- File size reduction analysis
- Pipeline efficiency optimization

**`braking_correlation.py`**: Statistical validation of brake signal extraction
- Correlation analysis between brake signal and speed gradient
- Cross-validation of OBD-II reverse engineering results

### Dataset Visualization

**`boxy_visualization.py`**: Boxy dataset analysis for YOLO fine-tuning validation
- Vehicle density heatmaps
- Class distribution analysis
- Suspicious annotation detection

---

## 🏗️ Evaluation Workflow

### 1. Model Training Phase
```bash
# Train single architecture
python training/multimodal/train_single.py --architecture simple_concat_transformer

# Monitor with Weights & Biases integration
```

### 2. Comprehensive Evaluation
```bash
# Single model evaluation
python evaluation/eval_single.py --model path/to/model

# Inference benchmarking
python evaluation/inference_benchmark.py --device cpu

# Attention analysis (Transformer only)
python evaluation/attention_analyzer.py --model path/to/transformer_model
```

### 3. Specialized Analysis
```bash
# False negative characteristics
python evaluation/analyze_false_negatives.py --model path/to/model

# YOLO performance baseline
python evaluation/performance.py --models 'yolo11n.pt, yolo12n.pt, yolo12n.mnn' --limit 100
```

---

## 📈 Key Metrics & Thresholds

### Primary Metrics (Extreme Imbalance)
- **PR-AUC**: Main metric for brake events (random baseline: 0.028)
- **ROC-AUC**: Secondary metric for discrimination capability
- **F1-Score**: At optimally determined thresholds

### Performance Baselines
- **Random Baseline**: 1.6% for brake events, 7.1% for coast events
- **Success Threshold**: PR-AUC > 2x random baseline
- **Best Achievement**: brake_1s PR-AUC = 0.203 (7.25x improvement over random)

### Hardware Constraints (Raspberry Pi 5)
- **Total Pipeline**: <133ms target
- **YOLO Component**: ~123ms (dominant bottleneck)
- **Multimodal Model**: <10ms (acceptable overhead)
- **Memory Budget**: <1.4GB total

---

## 🔧 Configuration & Reproducibility

### Focal Loss Configuration (Empirically Optimized)
```python
focal_loss_config = {
    "brake_1s": {"alpha": 0.2, "gamma": 3.0, "weight": 1.0},    # Safety-critical
    "brake_2s": {"alpha": 0.2, "gamma": 2.75, "weight": 0.95},
    "coast_1s": {"alpha": 0.3, "gamma": 2.0, "weight": 0.85},   # Efficiency
    "coast_2s": {"alpha": 0.3, "gamma": 1.75, "weight": 0.8}
}
```

### Evaluation Standards
- **Batch Size**: 256 for training, 64 for evaluation, 1 for inference benchmarks
- **Sequence Length**: 20 frames (10 seconds at 2Hz)
- **Stride Optimization**: Stride=1 optimal due to limited data availability
- **Cross-Validation**: Temporal split (70.4% train / 19.5% val / 10.1% test)

---

## 🎯 Scientific Contributions Validated

### 1. **Modular Architecture Evaluation**
Systematic comparison of Encoder/Fusion/Decoder combinations enables ablation studies and identifies optimal configurations for automotive constraints.

### 2. **Hardware-Aware Performance Analysis**
Realistic Raspberry Pi 5 benchmarks provide production-ready performance expectations, identifying YOLO as primary bottleneck and validating sub-10ms multimodal inference.

### 3. **Extreme Imbalance Handling**
Focal loss configuration and PR-AUC focus addresses realistic safety-critical event sparsity, with 7.25x improvement over random baseline for 1s brake prediction.

### 4. **Temporal Attention Insights**
Quantified recency bias validates hypothesis that local temporal context dominates long-range dependencies for brake prediction, supporting architecture choices.

---

## 📁 File Organization

```
evaluation/
├── eval_single.py              # 🎯 Main model evaluation
├── inference_benchmark.py      # ⚡ Performance benchmarking  
├── attention_analyzer.py       # 🔍 Transformer interpretability
├── analyze_false_negatives.py  # 🕵️ Intensity analysis
├── performance.py              # 📊 YOLO benchmarking
├── sequence_viewer.py          # 👁️ Interactive data explorer
├── plotters/                   # 📈 Data visualization utilities
│   ├── plot_data_rec.py        # Recording data plots
│   ├── plot_brake.py           # Brake gradient analysis
│   └── plot_gears.py           # Gear calculation validation
└── results/                    # 📁 Generated outputs and reports
```

Each script includes comprehensive documentation, scientific visualization capabilities, and JSON/CSV export functionality for further analysis and publication.