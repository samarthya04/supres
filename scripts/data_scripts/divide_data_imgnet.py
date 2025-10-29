import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from tqdm import tqdm

def copy_file(src_path: str, dest_dir: str):
    """Copies a single file to a destination directory."""
    try:
        shutil.copy(src_path, dest_dir)
    except Exception as e:
        print(f"Error copying {src_path}: {e}")

def process_and_resize_image(src_path: str, lr_dir: str, hr_dir: str, scale: int):
    """Copies the original image to HR dir and saves a resized version to LR dir."""
    try:
        # Copy original to HR folder
        shutil.copy(src_path, hr_dir)

        # Resize for LR folder
        file_name = os.path.basename(src_path)
        lr_path = os.path.join(lr_dir, file_name)
        with Image.open(src_path) as img:
            # Ensure image is in RGB format
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            new_width = img.width // scale
            new_height = img.height // scale
            
            # Ensure dimensions are at least 1x1
            if new_width < 1 or new_height < 1:
                # print(f"Skipping {src_path} due to small dimensions after scaling.")
                # If we skip, we should remove the HR copy as well
                os.remove(os.path.join(hr_dir, file_name))
                return

            img_resized = img.resize((new_width, new_height), Image.BICUBIC)
            img_resized.save(lr_path)
    except FileNotFoundError:
        # print(f"Warning: Source file not found {src_path}")
        pass
    except Exception as e:
        print(f"Error processing {src_path}: {e}")


def process_split(split_name: str, base_original_dir: str, base_target_dir: str, scales: list, max_workers: int):
    """
    Processes a dataset split (train, val, test) by copying HR images
    and creating downscaled LR images.
    """
    print(f"\nProcessing '{split_name}' split...")
    source_dir = os.path.join(base_original_dir, split_name)
    
    if not os.path.isdir(source_dir):
        print(f"Warning: Source directory not found for '{split_name}' split: {source_dir}")
        return

    # Find all JPEG files recursively
    image_paths = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.jpeg', '.jpg')):
                image_paths.append(os.path.join(root, file))

    if not image_paths:
        print(f"No images found for '{split_name}' split in {source_dir}.")
        return

    for scale in scales:
        print(f"  Generating HR/LR pairs for scale X{scale}...")
        hr_dir = os.path.join(base_target_dir, split_name, f"X{scale}", "HR")
        lr_dir = os.path.join(base_target_dir, split_name, f"X{scale}", "LR")
        os.makedirs(hr_dir, exist_ok=True)
        os.makedirs(lr_dir, exist_ok=True)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            with tqdm(total=len(image_paths), desc=f"  {split_name} X{scale}") as pbar:
                futures = [
                    executor.submit(process_and_resize_image, src_path, lr_dir, hr_dir, scale)
                    for src_path in image_paths
                ]
                for future in futures:
                    future.add_done_callback(lambda p: pbar.update(1))

def main():
    """
    Main function to organize the ImageNet dataset into HR/LR pairs for different scales.
    """
    base_original_dir = "ILSVRC2017_DET/ILSVRC/Data/DET"
    base_target_dir = "data/imagenet"
    
    # Check if the source directory exists
    if not os.path.isdir(base_original_dir):
        print(f"Error: The source ImageNet directory does not exist at '{base_original_dir}'.")
        print("Please ensure the dataset is downloaded and extracted according to the specified structure.")
        return

    splits_to_process = ["train", "val"] 
    scale_factors = [2, 4, 8]
    max_workers = os.cpu_count() or 1

    for split in splits_to_process:
        process_split(split, base_original_dir, base_target_dir, scale_factors, max_workers)
        
    print("\nDataset processing complete.")
    print(f"Processed data is located in: '{base_target_dir}'")

if __name__ == "__main__":
    main()
