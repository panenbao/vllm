import re
from collections import defaultdict

def calculate_layer_avg_time(log_file):
    """计算每层的平均执行时间"""
    layer_times = defaultdict(list)
    
    # 从日志文件读取数据
    with open(log_file, 'r') as f:
        for line in f:
            # 使用正则表达式提取层数和时间
            match = re.match(r'layer: (\d+), time: ([\d.]+)', line.strip())
            if match:
                layer, time = match.groups()
                layer = int(layer)
                time = float(time)
                layer_times[layer].append(time)
    
    # 计算每层的平均时间
    avg_times = {}
    for layer, times in layer_times.items():
        avg_times[layer] = sum(times) / len(times)
    
    # 按层数排序并打印结果
    print("\n层级平均执行时间统计:")
    print("-" * 40)
    print(f"{'层级':^10}{'平均时间(秒)':^15}{'采样数':^10}")
    print("-" * 40)
    
    for layer in sorted(avg_times.keys()):
        avg_time = avg_times[layer]
        samples = len(layer_times[layer])
        print(f"{layer:^10}{avg_time:^15.6f}{samples:^10}")

if __name__ == "__main__":
    log_file = "/home/panenbao/vllm/logs/time_per_layer.log"  # 替换为实际的日志文件路径
    calculate_layer_avg_time(log_file)
    log_file = "/home/panenbao/results/time_per_layer.log"  # 替换为实际的日志文件路径
    calculate_layer_avg_time(log_file)