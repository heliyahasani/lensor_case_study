# Lensor Case Study API

This repository contains the API for the Lensor case study. The API is built using FastAPI and provides endpoints for health checks and image inference.

## Getting Started

### Prerequisites

### Installation

Reposorities intial setup should be followed.

### Running the API

```sh
make run
```

1. API Endpoints

#### Health Check

- Endpoint: /healthcheck
- Method: GET
- Description: Returns a "Success!" message to indicate that the API is healthy.

#### Image Inference

- Endpoint: /inference
- Method: POST
- Description: Performs inference on an uploaded image.
- Request Body:
- file: An image file (e.g., JPEG, PNG).
  Response Body:
- A JSON object containing the inference results.

### Usage

#### Health Check

```sh
curl http://localhost:8087/healthcheck
```

#### Image Inference

```sh
curl -X POST http://localhost:8000/inference -F file=@path/to/image.jpg
```

### UI

For better user experience with the model you can use the UI provided.

1. Copy below url to the browser.

```sh

```
