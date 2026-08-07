# main.py
# Created by Brendan Fullerton on 2024-06-15.

# Import statements
import os
from pathlib import Path
import argparse
import pycolmap

def main(scene):
    # Get scene files
    print(f"Processing scene: {scene}")
    scene_folder = Path(f"./dataset/{scene}")
    output_dir = Path(f"./colmap_output/{scene}")
    train_dir = scene_folder / "train"
    test_dir = scene_folder / "test"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create COLMAP database
    database_path = output_dir / "database.db"
    print(f"Creating COLMAP database at: {database_path}")

    # Feature extraction
    print("Extracting features...")
    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.num_threads = -1
    pycolmap.extract_features(database_path=database_path,
                              image_path=train_dir,
                              camera_mode=pycolmap.CameraMode.AUTO,
                              extraction_options=extraction_options)

    # Feature matching
    print("Matching features...")
    matching_options = pycolmap.FeatureMatchingOptions()
    matching_options.use_gpu = True
    pycolmap.match_sequential(database_path=database_path,
                                matching_options=matching_options)

    # Sparse reconstruction
    print("Performing sparse structure-from-motion...")
    reconstructions = pycolmap.incremental_mapping(database_path=database_path,
                                        image_path=train_dir,
                                        output_path=output_dir)

    # Check results
    if reconstructions:
        print(f"Sparse reconstruction completed. Results saved in: {output_dir}")
        for index, sparse_model in reconstructions.items():
            print(f"Model {index}:")
            print(f"  Number of registered images: {len(sparse_model.images)}")
            print(f"  Number of 3D points: {len(sparse_model.points3D)}")

        # Save camera poses and point cloud - add later
    else:
        print("Sparse reconstruction failed. Please check the input data and parameters.")

    # Dense reconstruction (MVS) - add later



if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="The script for the Gaussian Splatting project.")

    # Define expected arguments
    parser.add_argument("--scene", type=str, required=True, default="scene1", help="Path to the scene directory.")

    # Parse inputs
    args = parser.parse_args()

    # Pass the parsed arguments to the main function
    main(scene=args.scene)