## Files Description

- **images/**: Contains subdirectories for training, validation, and test images.
- **annotations/**: Contains CSV files with annotations for training, validation, and test images.
- **merged/**: Contains merged CSV files that combine image data with annotations and category information.
- **notebooks/**: Contains a Jupyter notebook for interactive data exploration.

## Annotations CSV Format

Each annotations CSV file (`train_annotations.csv`, `validation_annotations.csv`, `test_annotations.csv`) contains the following columns:

- **file_name**: Name of the image file.
- **category_id**: ID of the damage category.
- **bbox**: Bounding box coordinates in the format `[x1, y1, width, height]`.
- **name**: Name of the damage category.
- **x1**: Top-left x-coordinate of the bounding box.
- **y1**: Top-left y-coordinate of the bounding box.
- **x2**: Bottom-right x-coordinate of the bounding box (calculated as x1 + width).
- **y2**: Bottom-right y-coordinate of the bounding box (calculated as y1 + height).
