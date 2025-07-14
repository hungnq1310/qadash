# 🤖 Qwen2-VL QA System

A Vietnamese intelligent Question-Answering system powered by Qwen2-VL multimodal model with RAG (Retrieval-Augmented Generation) capabilities. This system can analyze both textual data (CSV files) and visual content (images/dashboards) to provide comprehensive answers.

## ✨ Features

- 🔍 **Intelligent Question Classification**: Automatically determines whether to analyze images or data based on question content
- 📊 **CSV Data Analysis**: Load and query business data from CSV files
- 🖼️ **Image Analysis**: Analyze dashboards, charts, and visual content using Qwen2-VL vision capabilities
- 🧠 **RAG Integration**: Uses LangChain for retrieval-augmented generation with FAISS vector store
- 🌐 **Web Interface**: Beautiful Gradio UI for easy interaction
- 🇻🇳 **Vietnamese Support**: Optimized for Vietnamese language queries and responses

## 🏗️ Architecture

The system consists of two main components:

1. **Core Qwen2-VL Integration** (`main.py`): Basic model loading and inference
2. **Advanced QA System** (`gradio_ui.py`): Full-featured system with:
   - Custom LangChain LLM wrapper for Qwen2-VL
   - Vector store for document retrieval
   - Smart question routing
   - Web interface

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ GPU memory for Qwen2-VL-2B model

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd qadash
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the basic example:
```bash
python main.py
```

4. Launch the web interface:
```bash
python gradio_ui.py
```

The web interface will be available at `http://localhost:7860`

## 📁 Project Structure

```
qadash/
├── main.py              # Basic Qwen2-VL example
├── gradio_ui.py         # Full QA system with web UI
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── test.md             # Sample market report
└── data/
    ├── daily_report_20250628.csv  # Business data sample
    ├── demotest1.csv              # Raw data sample
    └── test1.png                  # Dashboard image sample
```

## 💡 Usage Examples

### Basic Model Usage

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Load model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", 
    torch_dtype="auto", 
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# Analyze image with text
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": "path/to/image.jpg"},
        {"type": "text", "text": "Describe this image."}
    ]
}]
```

### Web Interface Usage

1. **Upload Data**: Load CSV files containing business data
2. **Upload Images**: Upload dashboard screenshots or charts
3. **Ask Questions**: Type questions in Vietnamese or English
4. **Get Intelligent Answers**: The system automatically routes questions to appropriate analysis methods

### Question Types

**Image Analysis Questions** (routed to vision model):
- "Mô tả ảnh này"
- "Biểu đồ hiển thị gì?"
- "Màu sắc nào được sử dụng trong dashboard?"

**Data Analysis Questions** (routed to RAG system):
- "Tổng doanh thu là bao nhiêu?"
- "BU nào có volume cao nhất?"
- "Phân tích dữ liệu theo tháng"

## 🔧 Configuration

### Model Configuration

The system uses Qwen2-VL-2B-Instruct by default. You can modify the model in `gradio_ui.py`:

```python
# For better performance with more GPU memory
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",  # Larger model
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
```

### Vector Store Configuration

Adjust embedding model and chunk settings:

```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
```

## 📊 Data Format

### CSV Files
The system expects CSV files with business data. Sample structure:
- `daily_report_20250628.csv`: Daily business reports
- `demotest1.csv`: Raw transaction data

### Images
Supported formats: PNG, JPG, JPEG
- Dashboard screenshots
- Charts and graphs
- Business visualizations

## 🛠️ Development

### Adding New Features

1. **Custom Question Classifiers**: Modify `classify_question()` in `Qwen2VLLLM` class
2. **New Data Sources**: Extend `load_initial_data()` method
3. **UI Components**: Add new Gradio components in `create_interface()`

### Performance Optimization

- Use `flash_attention_2` for faster inference
- Adjust `min_pixels` and `max_pixels` for vision processing
- Implement batch processing for multiple queries

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size
   - Use smaller model variant
   - Enable gradient checkpointing

2. **Slow Inference**:
   - Enable flash attention
   - Use appropriate `torch_dtype`
   - Optimize image resolution

3. **Vietnamese Text Issues**:
   - Ensure UTF-8 encoding
   - Check tokenizer compatibility

## 📋 Requirements

### Core Dependencies
- `transformers>=4.37.0`
- `torch>=2.0.0`
- `qwen-vl-utils`
- `gradio`
- `langchain`
- `sentence-transformers`
- `faiss-cpu`

### Optional Dependencies
- `flash-attn` (for better performance)
- `accelerate` (for model loading)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

