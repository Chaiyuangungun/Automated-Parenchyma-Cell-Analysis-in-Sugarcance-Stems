import csv
import cv2
import numpy as np
import os
from skimage.filters import gaussian
from sklearn.cluster import KMeans
from cellpose import models
import multiprocessing
import argparse

# 设置命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Process images for clustering and cellpose segmentation.")
    parser.add_argument('-i', '--input', type=str, required=True, help="Input image folder path")
    parser.add_argument('-o', '--output', type=str, required=True, help="Output CSV file path")
    parser.add_argument('-b', '--bg_processes', type=int, default=8, help="Number of processes for background filtering and clustering")
    parser.add_argument('-c', '--cellpose_processes', type=int, default=64, help="Number of processes for Cellpose cell counting")
    return parser.parse_args()

# 获取命令行参数
args = parse_args()

# 输入图片文件夹路径
image_folder = args.input  # 从命令行获取文件夹路径

# 获取所有图片文件路径
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]

# 设置背景过滤时的块大小
background_block_size = 320
clustering_block_size = 160

# 设置聚类前的进程数和Cellpose计数的进程数
bg_processes = args.bg_processes  # 从命令行获取背景过滤和聚类的进程数
cellpose_processes = args.cellpose_processes  # 从命令行获取Cellpose细胞计数的进程数

# 聚类步骤的函数
def process_image_for_clustering(img_path):
    print("running "+img_path)
    image_path = os.path.join(image_folder, img_path)
    img = cv2.imread(image_path)

    # 检查图像是否正确加载
    if img is None:
        print(f"Image {img_path} could not be loaded.")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 获取图像的高度和宽度
    height, width, _ = img_rgb.shape
    num_blocks_vertical_bg = height // background_block_size
    num_blocks_horizontal_bg = width // background_block_size

    # 过滤掉白色背景的块
    valid_mask = np.ones((num_blocks_vertical_bg, num_blocks_horizontal_bg), dtype=bool)

    # 白色阈值，判断是否是白色
    white_threshold = 245

    # 遍历图像中的每个块，识别背景区域
    for i in range(num_blocks_vertical_bg):
        for j in range(num_blocks_horizontal_bg):
            block = img_rgb[i * background_block_size:(i + 1) * background_block_size,
                            j * background_block_size:(j + 1) * background_block_size]

            block_gray = cv2.cvtColor(block, cv2.COLOR_RGB2GRAY)
            white_pixels = np.sum(block_gray > white_threshold)
            total_pixels = background_block_size * background_block_size

            white_ratio = white_pixels / total_pixels
            if white_ratio >= 0.9:
                valid_mask[i, j] = False

    # -------------------------------------------背景过滤后的第二步聚类-------------------------------------------

    num_blocks_vertical_clustering = height // clustering_block_size
    num_blocks_horizontal_clustering = width // clustering_block_size

    color_matrix = np.zeros((num_blocks_vertical_clustering, num_blocks_horizontal_clustering, 3), dtype=float)

    # 遍历图像中的每个块，计算每个块的平均颜色
    for i in range(num_blocks_vertical_clustering):
        for j in range(num_blocks_horizontal_clustering):
            # 获取当前块的区域
            block = img_rgb[i*clustering_block_size:(i+1)*clustering_block_size, 
                            j*clustering_block_size:(j+1)*clustering_block_size]
        
            # 计算块的平均颜色（RGB空间）
            avg_color = np.mean(block, axis=(0, 1))  # 在x, y轴上计算均值
            color_matrix[i, j] = avg_color

    # 对颜色矩阵进行平滑处理
    smooth_color_matrix = gaussian(color_matrix, sigma=1)

    # 扩展 valid_mask 的形状，使其与 smooth_color_matrix 匹配
    scaling_factor = background_block_size // clustering_block_size
    valid_mask_expanded = np.repeat(np.repeat(valid_mask, scaling_factor, axis=0), scaling_factor, axis=1)

    # 确保扩展后的掩码形状与 smooth_color_matrix 匹配
    valid_mask_expanded = valid_mask_expanded[:smooth_color_matrix.shape[0], :smooth_color_matrix.shape[1]]

    # 获取有效区域的像素
    valid_pixels = smooth_color_matrix[valid_mask_expanded]

    if valid_pixels.size == 0:
        print(f"No valid pixels found for clustering in image: {img_path}")
        return None

    # 聚类
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(valid_pixels)

    # 获取每个像素的聚类标签
    labels = kmeans.labels_

    classified_matrix = np.full(valid_mask_expanded.shape, -1, dtype=int)
    classified_matrix[valid_mask_expanded] = labels

    cluster_centers = kmeans.cluster_centers_
    brightness = np.sum(cluster_centers, axis=1)
    bright_class = np.argmax(brightness)
    dark_class = 1 - bright_class

    return img_rgb, classified_matrix, bright_class, dark_class


# 使用多进程进行背景过滤和聚类
def process_all_images(images):
    with multiprocessing.Pool(bg_processes) as pool:
        results = pool.map(process_image_for_clustering, images)

    # 结果存储到字典中，跳过返回 None 的结果
    clustering_results = {}
    for img_path, result in zip(images, results):
        if result is not None:
            img_rgb, classified_matrix, bright_class, dark_class = result
            clustering_results[img_path] = {
                'img_rgb': img_rgb,
                'classified_matrix': classified_matrix,
                'bright_class': bright_class,
                'dark_class': dark_class
            }
    return clustering_results


# Cellpose细胞计数函数
def cellpose_segmentation_and_count(img_rgb, classified_matrix_rescaled, bright_class, dark_class):
    model = models.Cellpose(gpu=False, model_type='cyto2')
    masks, *_ = model.eval(img_rgb, diameter=None, channels=[0, 0], min_size=6400, cellprob_threshold=2)

    scaling_factor_for_masks = img_rgb.shape[0] // classified_matrix_rescaled.shape[0]
    classified_matrix_rescaled = np.repeat(np.repeat(classified_matrix_rescaled, scaling_factor_for_masks, axis=0), scaling_factor_for_masks, axis=1)

    def count_cells_in_class(masks, class_mask, threshold=0.5):

        unique_cells = np.unique(masks[class_mask > 0])  # 获取所有非背景的细胞标签
        unique_cells = unique_cells[unique_cells > 0]   # 排除背景标签（通常为0）
    
        count = 0
        for cell in unique_cells:
            cell_mask = (masks == cell)  # 获取该细胞的掩码
            # 计算该细胞区域在指定类别中的占比
            cell_in_class = np.sum(cell_mask & (class_mask > 0))  # 该细胞与类别区域重叠的像素数
            cell_area = np.sum(cell_mask)  # 该细胞的总像素数
            cell_ratio = cell_in_class / cell_area  # 计算占比
        
            # 如果占比超过阈值，则计入该类别
            if cell_ratio >= threshold:
                count += 1
            
        return count
    bright_class_cells = count_cells_in_class(masks, (classified_matrix_rescaled == bright_class))
    dark_class_cells = count_cells_in_class(masks, (classified_matrix_rescaled == dark_class))
    invalid_class_cells = count_cells_in_class(masks, (classified_matrix_rescaled == -1))

    bright_class_area = np.sum(masks == bright_class)
    dark_class_area = np.sum(masks == dark_class)

    return bright_class_cells, dark_class_cells, invalid_class_cells, bright_class_area, dark_class_area


# 处理所有图片并计算细胞
def process_cells_for_all_images(clustering_results):
    with multiprocessing.Pool(cellpose_processes) as pool:
        results = [
            pool.apply_async(cellpose_segmentation_and_count, (result['img_rgb'], result['classified_matrix'], result['bright_class'], result['dark_class']))
            for result in clustering_results.values()
        ]
        pool.close()
        pool.join()

        cell_count_results = {}
        for img_path, result in zip(clustering_results.keys(), results):
            bright_class_cells, dark_class_cells, invalid_class_cells, bright_class_area, dark_class_area = result.get()
            cell_count_results[img_path] = {
                'bright_class_cells': bright_class_cells,
                'dark_class_cells': dark_class_cells,
                'invalid_class_cells': invalid_class_cells,
                'bright_class_area': bright_class_area,
                'dark_class_area': dark_class_area
            }

    return cell_count_results


# 获取聚类结果
clustering_results = process_all_images(image_files)

# 获取细胞计数结果
cell_count_results = process_cells_for_all_images(clustering_results)

# 将结果输出到CSV文件
output_file = args.output  # 从命令行获取输出文件路径
with open(output_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    # 写入表头
    writer.writerow(['Image', 'Bright Class Cells', 'Dark Class Cells', 'Invalid Class Cells', 'Bright Class Area', 'Dark Class Area'])
    
    # 写入每个图像的结果
    for img_path, counts in cell_count_results.items():
        writer.writerow([img_path, counts['bright_class_cells'], counts['dark_class_cells'], counts['invalid_class_cells'], counts['bright_class_area'], counts['dark_class_area']])

print(f"Results have been written to {output_file}")
