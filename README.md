# Baby Sonnet Quest: Train Your Own Mini-LLM

A gamified, interactive Python tutorial that teaches the fundamentals of training language models using PyTorch.

This project is designed for beginners who want to understand how Large Language Models (LLMs) like Claude work under the hood. It breaks down complex concepts—tokenization, model architecture, loss calculation, and backpropagation—into a text-based adventure game.

## Features

- **Interactive Levels**: Progress through 4 levels: Data Prep, Model Architecture, Training, and Generation.
- **Visual Feedback**: Watch the training loss decrease in real-time.
- **Customizable Architectures**: Choose between RNN, GRU, or LSTM models.
- **Zero-Setup Data**: Automatically downloads the Tiny Shakespeare dataset.

## Quick Start

### Prerequisites
- Python 3.8+
- PyTorch

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/baby-sonnet-quest.git
    cd baby-sonnet-quest
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### How to Play

Run the game script:
```bash
python3 quest.py
```

Follow the on-screen instructions to:
1.  Load and tokenize the dataset.
2.  Select your neural network architecture.
3.  Watch the training process (the "Training Ritual").
4.  Generate new text in the style of Shakespeare!

## What You'll Learn
- **Tokenization**: Converting text to integer sequences.
- **Embeddings**: Representing tokens as vectors.
- **Sequence Modeling**: How RNNs/LSTMs process sequential data.
- **Optimization**: The forward/backward pass loop.
- **Sampling**: Generating text from probabilities.

## License
MIT License
