Food Classification with Transfer Learning
A deep learning project for food image classification using transfer learning techniques. The model achieves approximately 0.8 on various metrics.


🌟 Features
Image classification using transfer learning
Interactive visualizations with Plotly
Comprehensive experiment logging
Data preprocessing pipeline


📁 Project Structure
.
├── main.py              # Main execution script
└── src/                 # Source code modules
    ├── preprocessor.py  # Data preprocessing pipeline
    ├── plotlies.py     # Visualization functions
    └── logexp.py       # Logging and experiment tracking

🔧 Components
main.py
Main script for running the classification pipeline. Coordinates the preprocessing, training, and evaluation processes.

src/preprocessor.py
Image loading and dataset preparation for training

src/plotlies.py
Visualization module using Plotly:
    class distribution pieplot
    Training metrics visualization
    Model performance analysis

src/logexp.py
Manages experiment logging and tracking:
    Training progress monitoring
    Performance metrics logging
    Experiment configuration tracking


📊 Results
The model achieves approximately 0.8 across evaluation metrics, with specific considerations:
    Strong performance on most food categories
    Some confusion between visually similar items (e.g., tacos, taquitos)


🔄 Future Improvements
Data Collection Enhancements:
Improved photography guidelines for better feature visibility
Standardized image capture procedures

Model Architecture:
Two-stage classification system for similar food items
Fine-tuning of transfer learning parameters


🚀 Getting Started
Clone the repository:
!git clone https://github.com/Giovannicus/GourmetAI.git

Or save in .py and run the main script:
python main_notyebook.py


📈 Model Performance
Overall Accuracy: ~0.8
Detailed visualizations available through the Plotly interface
Performance analysis accessible via logging module

👥 Contact
All About Me On My Git
