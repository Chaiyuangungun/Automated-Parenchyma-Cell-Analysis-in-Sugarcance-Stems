import os
import argparse
import openslide
from PIL import Image
from multiprocessing import Pool

def process_svs_file(args):
    svs_file, input_dir, output_dir, tile_size = args

    if not svs_file.endswith(".svs"):
        return f"跳过无效文件: {svs_file}"

    svs_path = os.path.join(input_dir, svs_file)

    # 打开 SVS 文件
    try:
        slide = openslide.OpenSlide(svs_path)
    except Exception as e:
        return f"无法打开文件 {svs_file}: {e}"

    # 获取文件名并创建对应的输出子目录
    base_name = os.path.splitext(svs_file)[0]
    image_output_dir = os.path.join(output_dir, base_name)
    os.makedirs(image_output_dir, exist_ok=True)

    # 获取 SVS 文件的宽和高
    width, height = slide.dimensions

    # 按网格切割图片并保存
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tile = slide.read_region((x, y), 0, (tile_size, tile_size))
            tile = tile.convert("RGB")  # 转为 JPEG 支持的 RGB 格式
            output_path = os.path.join(image_output_dir, f"tile_{x}_{y}.jpg")
            tile.save(output_path, "JPEG")

    return f"文件 {svs_file} 的切割完成，结果保存在 {image_output_dir}"

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="切割 SVS 文件成小块并保存为 JPG")
    parser.add_argument("-i", "--input", type=str, required=True, help="输入 SVS 文件所在目录")
    parser.add_argument("-o", "--output", type=str, required=True, help="输出图片保存目录")
    parser.add_argument("-t", "--process", type=int, default=4, help="并行处理的进程数（默认 4）")
    parser.add_argument("-s", "--size", type=int, default=2560, help="切割图片的大小（默认 2560x2560）")

    args = parser.parse_args()
    input_dir = args.input
    output_dir = args.output
    num_processes = args.process
    tile_size = args.size

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有 SVS 文件
    svs_files = [f for f in os.listdir(input_dir) if f.endswith(".svs")]

    # 使用多进程处理 SVS 文件
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_svs_file, [(svs_file, input_dir, output_dir, tile_size) for svs_file in svs_files])

    # 输出处理结果
    for result in results:
        print(result)

if __name__ == "__main__":
    main()
