# Automated-Parenchyma-Cell-Analysis-in-Sugarcance-Stems
  To process high-resolution SVS (Scanned Virtual Slide) files, we implemented a method for cell Segmentation and Counting. This workflow including three steps: 
(1) Image tiling and conversion
(2) Background filtering and vascular bundle structures identification
(3) Cell segmentation and counting: This step employs the Cellpose  to perform segmentation of cellular structures within an image. The cyto2 model is used with parameters a minimum cell area of “min_size = 6400" and a cell probability threshold of “cellprob_threshold = 2" to ensure accurate detection of parenchymal cells. Finally, the number of parenchymal cells is counted based on the regions identified as parenchymal cells in step (2).
# Dependencies
Python Modules
  pip install opencv-python numpy scikit-image scikit-learn cellpose Pillow
# Usage
  1. SVS File Tiling Script: Convert Whole Slide Images to JPEG Tiles
     
     Prepare the Input Directory:
          Place all .svs files in a directory.
  
    python3 svs2jpg.py -i <input_directory> -o <output_directory> -t <num_processes> -s 2560
      -i / --input (Required): The directory containing the SVS files to be processed.
      -o / --output (Required): The directory where the resulting image tiles will be saved.
      -t / --process (Optional, default: 4): The number of processes to use for parallel processing.      
      -s / --size (Optional, default: 2560): The size of each tile (in pixels).
      
  2. Parenchyma Cell Analysis in Sugarcane Stems

    python PCanalysis.py -i /path/to/image_folder -o /path/to/output.csv -b 8 -c 64
      -i / --input (Required): The directory containing the image files to be processed.
      -o / --output (Required): The path where the output CSV file will be saved.
      -b / --bg_processes (Optional, default: 8): The number of processes to use for background filtering and clustering.
      -c / --cellpose_processes (Optional, default: 64): The number of processes to use for Cellpose cell counting.      
  3.result
      
      jpg.id, Parenchyma Cell count, Others count, Parenchyma Cell size, Others size
    
    
