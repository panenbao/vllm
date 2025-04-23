import re
from collections import defaultdict

def parse_log_file(file_path):
    """解析日志文件，返回每层的执行时间列表"""
    layer_times = defaultdict(list)
    with open(file_path, 'r') as f:
        for line in f:
            match = re.match(r'layer: (\d+), time: ([\d.]+)', line.strip())
            if match:
                layer, time = match.groups()
                layer = int(layer)
                time = float(time)
                layer_times[layer].append(time)
    return layer_times

def compare_log_files(file1_path, file2_path):
    """比较两个日志文件的执行时间差异"""
    times1 = parse_log_file(file1_path)
    times2 = parse_log_file(file2_path)
    
    # 获取所有层级
    all_layers = sorted(set(times1.keys()) | set(times2.keys()))
    
    # 打印比较结果
    print("\n日志文件执行时间对比:")
    print("-" * 80)
    print(f"{'层级':^8}{'文件1平均时间':^15}{'文件2平均时间':^15}{'差异':^15}{'差异百分比':^15}{'采样数':^12}")
    print("-" * 80)
    
    for layer in all_layers:
        # 计算平均时间
        avg1 = sum(times1[layer]) / len(times1[layer]) if layer in times1 else 0
        avg2 = sum(times2[layer]) / len(times2[layer]) if layer in times2 else 0
        
        # 计算差异
        diff = avg2 - avg1
        diff_percent = (diff / avg1 * 100) if avg1 != 0 else float('inf')
        
        # 获取采样数
        samples1 = len(times1[layer]) if layer in times1 else 0
        samples2 = len(times2[layer]) if layer in times2 else 0
        
        print(f"{layer:^8}"
              f"{avg1:^15.6f}"
              f"{avg2:^15.6f}"
              f"{diff:^15.6f}"
              f"{diff_percent:^15.2f}%"
              f"{samples1}/{samples2:^8}")

if __name__ == "__main__":
    # 替换为实际的日志文件路径
    log_file1 = "/home/panenbao/vllm/logs/time_per_layer.log"
    log_file2 = "/home/panenbao/results/time_per_layer.log"
    
    compare_log_files(log_file1, log_file2)