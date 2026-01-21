# Target Tracking System with LLM-Enhanced Analysis

A comprehensive target tracking system that demonstrates various tracking algorithms and incorporates large language model (LLM) analysis for intelligent pattern recognition and anomaly detection.
<img width="5370" height="4719" alt="wrong_model_with_llm_tracking" src="https://github.com/user-attachments/assets/847965ee-532d-4061-a764-6f1d4130f93e" />


## Overview

This project implements multiple tracking algorithms including Kalman filters and Interactive Multiple Model (IMM) filters, enhanced with LangGraph-based LLM reasoning for semantic understanding of tracking behavior. The system provides both traditional statistical analysis and natural language explanations of model performance.

## Key Features

- **Multiple Tracking Algorithms**: Static Kalman, IMM with 2/3 models
- **LLM-Enhanced Analysis**: Local LLM integration via Ollama for pattern recognition
- **Structured Output**: Pydantic-validated JSON responses from LLM
- **LangGraph Workflows**: Multi-node analysis pipelines with decision fusion
- **Real-time Monitoring**: Innovation sequence analysis with confidence scoring
- **Visualization**: Comprehensive plotting of tracking performance and metrics

## Project Structure

```
cc_test/
├── base/                        # Core tracking algorithms
│   ├── motion.py                # Motion simulation and target dynamics
│   ├── trackers.py              # Kalman filter implementations
│   ├── imm.py                   # Interactive Multiple Model filters
│   └── simulators.py            # Simulation utilities
├── demo/                        # Demonstration scripts
│   ├── tracking_1_model.py      # Single Kalman filter demo
│   ├── tracking_imm_2_model.py  # Two-model IMM demo
│   ├── tracking_imm_3_model.py  # Three-model IMM demo
│   ├── tracking_langgraph_logic.py          # Logic-based LangGraph analysis
│   └── tracking_langgraph_llm_reasoning.py  # LLM-enhanced tracking
├── results/                     # Output plots and analysis results
└── pyproject.toml               # Project dependencies
```

## Prerequisites

### System Requirements

- Python 3.11 or higher
- Ollama (for LLM functionality)

### Ollama Installation and Setup

1. **Install Ollama**
   ```bash
   # On Linux/macOS
   curl -fsSL https://ollama.ai/install.sh | sh

   # On Windows, download from https://ollama.ai/download
   ```

2. **Start Ollama Service**
   ```bash
   # Start the Ollama service (runs on localhost:11434 by default)
   ollama serve
   ```

3. **Install TinyLlama Model**
   ```bash
   # Pull the TinyLlama model for fast inference
   ollama pull tinyllama

   # Verify installation
   ollama list
   ```

4. **Test Ollama Installation**
   ```bash
   # Test basic functionality
   ollama run tinyllama "Hello, how are you?"
   ```

### Python Environment Setup

1. **Clone and Navigate to Project**
   ```bash
   git clone <repository-url>
   cd cc_test
   ```

2. **Install Dependencies**
   ```bash
   # Using uv (recommended)
   uv sync

   # Or using pip
   pip install -e .
   ```

## Usage

### Basic Tracking Demonstrations

1. **Single Kalman Filter**
   ```bash
   python demo/tracking_1_model.py
   ```

2. **IMM with Two Models**
   ```bash
   python demo/tracking_imm_2_model.py
   ```

3. **IMM with Three Models**
   ```bash
   python demo/tracking_imm_3_model.py
   ```

### LLM-Enhanced Analysis

1. **Logic-Based LangGraph Analysis**
   ```bash
   python demo/tracking_langgraph_logic.py
   ```

2. **LLM-Enhanced Tracking with Structured Output**
   ```bash
   # Ensure Ollama is running with TinyLlama model
   python demo/tracking_langgraph_llm_reasoning.py
   ```

## LLM Integration Details

### Structured Output Schema

The system uses Pydantic models to ensure structured, validated responses from the LLM:

```python
class InnovationAnalysis(BaseModel):
    model_mismatch_detected: bool
    confidence_score: float = Field(ge=0.0, le=1.0)
    bias_severity: Literal["none", "low", "moderate", "high", "severe"]
    pattern_type: Literal["normal_noise", "systematic_bias", "trend_drift", "oscillatory", "unknown"]
    recommended_action: Literal["continue_current", "switch_to_cv", "switch_to_ca", "hybrid_approach", "investigate"]
    brief_explanation: str = Field(max_length=200)
```

### LangGraph Workflow

The LLM-enhanced system uses a multi-node workflow:

1. **Statistical Analysis**: Traditional innovation sequence analysis
2. **LLM Reasoning**: Semantic pattern recognition using TinyLlama
3. **Decision Fusion**: Combines statistical and semantic insights
4. **Alert Generation**: Produces actionable recommendations

## Configuration

### Ollama Model Configuration

The default configuration uses TinyLlama for fast inference. To change models:

```python
# In tracking_langgraph_llm_reasoning.py
ollama_model = "tinyllama"  # Change to other models like "llama3", "mistral", etc.
```

### Output Directories

Results are automatically saved to:

- `results/tracking_langgraph_llm_reasoning/` - LLM-enhanced tracking outputs
- `results/tracking_imm_2_model/` - IMM tracking results
- Other demo-specific directories

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   ```
   Error: Could not connect to Ollama at http://localhost:11434
   ```
   **Solution**: Ensure Ollama service is running (`ollama serve`)

2. **Model Not Found**
   ```
   Error: model 'tinyllama' not found
   ```
   **Solution**: Pull the model (`ollama pull tinyllama`)

3. **JSON Parsing Errors**
   ```
   Error: Unterminated string in JSON response
   ```
   **Solution**: The system includes fallback mechanisms for malformed LLM responses

4. **Confidence Score Validation Error**
   ```
   Error: confidence_score should be less than or equal to 1
   ```
   **Solution**: The system automatically normalizes percentage values to decimals

### Ollama Version Compatibility

- **Minimum Version**: Ollama 0.12.0 or higher
- **Recommended**: Latest stable version
- **Update Command**: Follow Ollama's official update instructions

## Performance Notes

- **TinyLlama**: Optimized for speed (~1-2 seconds per inference)
- **Structured Output**: Eliminates parsing errors through Pydantic validation
- **Batch Processing**: Supports analysis of long tracking sequences
- **Memory Usage**: Efficient handling of extended simulations

## Contributing

1. Follow existing code style and conventions
2. Test LLM integration with Ollama before submitting
3. Update documentation for new features
4. Ensure compatibility with current Ollama API versions

## License

[Add your license information here]
