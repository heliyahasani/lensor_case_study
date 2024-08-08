# Lensor Damage Detection Challenge

It is a deep neural network for object detection and finding damages on a vehicle.

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

## Installation and Setup

### Clone the repository

```sh
git clone https://github.com/yourusername/lensor_damage_detection.git
```

1. For install dependencies you need to have poetry installed.

- Navigate to the project directory and install the required dependencies using Poetry:

```sh
cd lensor_damage_detection
poetry shell
poetry install
```

### Use Docker Container

```sh
cd lensor_damage_detection
docker build -t lensor .
```

## Use Endpoint For Inference
