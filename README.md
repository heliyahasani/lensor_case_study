# Vehicle Damage Detection using Deep Learning

## Overview

This project aims to build a deep neural network to accurately detect damages on a vehicle given a single RGB image as input. The expected output is a set of points, the class, and the prediction confidence representing the detected damage.

## Solution Choices

I decided to use fasterrcnn_resnet50_fpn for object detection due to its balance between accuracy and speed. Additionally, fasterrcnn_resnet50_fpn has pre-trained weights available, which can help in achieving better results within the given time frame. PyTorch was chosen as the deep learning library due to its flexibility and wide adoption in the research community.

## Project Structure

- `dataset/`: Contains the dataset including annotations and images split into training, validation, and test sets.

  - `annotations/`: COCO format annotations for the images.
  - `images/`: Images categorized into `train`, `val`, and `test` directories.

- `src/`: Source code for the project.

  - `api/`: Code related to API endpoints.
  - `configs/`: Configuration files for the project.
  - `data_exploration/`: Notebooks and results related to data exploration.
  - `examples/`: Example scripts and code snippets.

- `tests/`: Contains test scripts for the project.

## Documentation

The code is well-documented with comments explaining the functionality of each part. Detailed instructions on how to set up and run the project are provided below.

## Commit History

The commit history is organized and meaningful, demonstrating the progress throughout the challenge. Key milestones and changes are clearly reflected in the commit messages.

## Visualizations

I used TensorBoard for visualizing training curves and sample images with detected damages. These visualizations help in understanding and interpreting the model's performance.

## Training Logs

Training logs, including metrics such as loss and accuracy, are included in the `logs/` directory. These logs provide insights into the model's training process and its performance over time.

## Inference Results on a Test Set

Due to time constraints, I was unable to achieve proper inference results. However, the model architecture and pipeline are set up correctly, and with further training, better results can be expected. Example inference images and their evaluation metrics are provided for reference.

## Testing

Unit tests for key functions are included to ensure the correctness of the code. These tests can be found in the `tests/` directory.

## Packaging/Deployment

The code is dockerized, and an endpoint for model inference is set up using FastAPI. However, the service inference is not working as expected due to insufficient training of the model. The Dockerfile and deployment instructions are provided to replicate the setup.

## Installation and Setup

### Clone the repository

```sh
git clone https://github.com/yourusername/lensor_damage_detection.git
```

1. For install dependencies you need to have poetry installed.

- Navigate to the project directory and install the required dependencies using Poetry:

```sh
cd lensor_damage_detection
make install
```

### Use Docker Container

```sh
cd lensor_damage_detection
make docker-build
make docker-run
```

## Challenges

- **Training**: Due to time constraints and the complexity of the task, the model could not be properly trained for inference. More training time and fine-tuning are required for better results.
- **Inference**: The current inference results are not satisfactory. Further work is needed to improve the detection accuracy.

## Conclusion

Despite the challenges faced, this project demonstrates a well-rounded pipeline for vehicle damage detection using deep learning. The model architecture is in place, and with additional training, it has the potential to achieve better performance. The code is well-documented, and the deployment setup is provided for ease of use.

## Future Work

- Improve the training process to achieve better inference results.
- Enhance the model architecture for more accurate damage detection.
- Add more tests to cover different scenarios and edge cases.
- Optimize the deployment for production use.

---

I hope this README provides a clear overview of the project, the work done, and the challenges faced. If you have any questions or need further assistance, please feel free to reach out.
