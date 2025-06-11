Project Requirements

1. Project Overview
Title: Image Restoration Model Using Restormer with Federated Learning
Description: Develop a federated learning system using the Restormer architecture to restore images degraded by haze, rain, or snow. The project will be implemented and managed using the Trae.ai IDE.
2. Technical Requirements
Programming Language & Libraries
Python 3.8+
PyTorch 2.0+ (core deep learning framework)
Torchvision (image utilities)
OpenCV (image processing)
Pillow (image handling)
NumPy (numerical operations)
Matplotlib (visualization)
tqdm (progress bars)
einops (tensor operations, required by Restormer)
timm (vision models, required by Restormer)
Flower (federated learning framework, PyTorch compatible)
Model Architecture
Restormer (Encoder-Decoder Transformer for image restoration)
Use official Restormer GitHub code and pretrained weights as needed.
3. Dataset Structure
Organize your data as follows for each degradation type (Haze, Rain, Snow):
text
Haze/
  Train/
     Input/    # Degraded images
     GT/       # Clean images
  Test/
     Input/    # Degraded images
     GT/       # Clean images

Snow/
  Train/
     Input/    # Degraded images
     GT/       # Clean images
  Test/
     Input/    # Degraded images
     GT/       # Clean images

Rain/
  Train/
     Input/    # Degraded images
     GT/       # Clean images
  Test/
     Input/    # Degraded images
     GT/       # Clean images
Note: Ensure all images are properly paired and preprocessed according to Restormer’s input requirements.
4. Federated Learning Setup
Framework: Flower (recommended for PyTorch)
Clients: 3 (one each for Haze, Rain, Snow)
Central Server: For model aggregation (FedAvg)
Secure Aggregation: Optional (for privacy)
Differential Privacy: Optional (for enhanced privacy)
5. Training & Evaluation
Batch size: 32
Learning rate: 3e-4 (AdamW optimizer recommended)
Loss function: Charbonnier Loss (as per Restormer paper)
Metrics:
PSNR (Peak Signal-to-Noise Ratio)
SSIM (Structural Similarity Index)
MSE (Mean Squared Error)
Metric Display and Reporting:
PSNR and SSIM values will be computed and displayed after each global round of federated learning, i.e., after the server aggregates model updates from all clients and evaluates the global model on the validation or test set.
These values are not typically shown after every local epoch on each client, but rather after each round of global aggregation.
A final evaluation and reporting of PSNR and SSIM values will be performed at the end of all training rounds to summarize the model’s restoration performance.
6. Checkpoints
What are checkpoints?
Checkpoints are saved snapshots of your model’s state (weights, biases, optimizer state, and sometimes training progress) during training. They allow you to resume training from the last saved point if interrupted, avoid losing progress, and keep the best-performing model for inference or deployment.
Why use checkpoints?
Resume training after interruptions or failures.
Save time and computational resources by not starting over.
Compare model performance at different training stages.
Deploy or fine-tune the best model version.
Implementation:
Use PyTorch’s torch.save() to create checkpoints at regular intervals or when the model achieves the best validation performance.
Store checkpoints in a dedicated directory (e.g., checkpoints/).
Restore training using torch.load() if needed.
Best Practice:
Save checkpoints when the model achieves its highest accuracy or lowest loss on validation data, not necessarily after every epoch, to conserve storage and focus on the best results.
7. Trae.ai IDE Configuration
PyTorch support enabled
Jupyter Notebooks or Python scripts for training
AI code assistance enabled
GitHub/GitLab integration for version control
Cloud collaboration tools (if working in a team)
8. Hardware Requirements
GPU: NVIDIA GPU with ≥16GB VRAM (Restormer is resource-intensive)
RAM: At least 16GB recommended
Storage: Sufficient for datasets, model checkpoints, and logs
9. Security & Privacy
Data privacy: No raw data leaves the client devices
Model security: Use secure aggregation and/or differential privacy if required
10. Documentation & Reporting
README.md: Project description, setup, and usage instructions
requirements.txt: All Python dependencies
Code comments: Especially for customizations to Restormer or FL logic
Evaluation reports: Include PSNR, SSIM, and visual comparisons for each degradation type
11. References
Restormer Paper
Restormer GitHub
Flower Federated Learning
[Checkpointing in Deep Learning]