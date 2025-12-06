# Contributing to Tourism Prediction

Thank you for your interest in contributing to the Tourism Prediction project!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/tourism-prediction.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Install dependencies: `pip install -r tourism_project/workflow_requirements.txt`

## Development Workflow

1. Make your changes
2. Test your changes locally:
   ```bash
   # Test data preparation
   python tourism_project/model_building/data_preparation.py
   
   # Test model training
   python tourism_project/model_building/model_training.py
   ```
3. Commit your changes with clear messages
4. Push to your fork
5. Create a Pull Request

## Code Style

- Follow PEP 8 guidelines
- Add docstrings to functions and classes
- Keep functions focused and modular
- Add comments for complex logic

## Testing

Ensure all scripts run without errors before submitting a PR:
- Data registration should handle missing HF_TOKEN gracefully
- Data preparation should create all necessary preprocessing files
- Model training should work with or without MLflow server

## Pull Request Process

1. Update README.md if you add features
2. Ensure your PR description clearly describes the changes
3. Link any related issues
4. Request review from maintainers

## Questions?

Open an issue for questions or discussions!
