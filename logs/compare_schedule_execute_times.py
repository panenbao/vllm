import re
import numpy as np

def parse_time_file(filename):
    time1_list = []
    time2_list = []
    
    with open(filename, 'r') as f:
        for line in f:
            # 修改后的正则表达式，可以匹配普通数字和科学计数法
            matches = re.findall(r'Time \d: ([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
            if len(matches) == 2:
                time1_list.append(float(matches[0]))
                time2_list.append(float(matches[1]))
    
    return np.array(time1_list), np.array(time2_list)

def format_time(value):
    """格式化时间值，对于很小的数使用科学计数法"""
    if abs(value) < 0.0001:
        return f"{value:.6e}"
    return f"{value:.6f}"

def analyze_times(times, name):
    return {
        'name': name,
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times)
    }

def compare_files(file1, file2):
    # 读取两个文件的数据
    time1_a, time2_a = parse_time_file(file1)
    time1_b, time2_b = parse_time_file(file2)
    
    # 分析数据
    print(f"\n文件 {file1} 的统计结果：")
    print("-" * 50)
    stats1_a = analyze_times(time1_a, "Time 1")
    stats2_a = analyze_times(time2_a, "Time 2")
    
    print(f"{stats1_a['name']}:")
    print(f"  平均值: {format_time(stats1_a['mean'])}")
    print(f"  标准差: {format_time(stats1_a['std'])}")
    print(f"  最小值: {format_time(stats1_a['min'])}")
    print(f"  最大值: {format_time(stats1_a['max'])}")
    
    print(f"\n{stats2_a['name']}:")
    print(f"  平均值: {format_time(stats2_a['mean'])}")
    print(f"  标准差: {format_time(stats2_a['std'])}")
    print(f"  最小值: {format_time(stats2_a['min'])}")
    print(f"  最大值: {format_time(stats2_a['max'])}")
    
    print(f"\n文件 {file2} 的统计结果：")
    print("-" * 50)
    stats1_b = analyze_times(time1_b, "Time 1")
    stats2_b = analyze_times(time2_b, "Time 2")
    
    print(f"{stats1_b['name']}:")
    print(f"  平均值: {format_time(stats1_b['mean'])}")
    print(f"  标准差: {format_time(stats1_b['std'])}")
    print(f"  最小值: {format_time(stats1_b['min'])}")
    print(f"  最大值: {format_time(stats1_b['max'])}")
    
    print(f"\n{stats2_b['name']}:")
    print(f"  平均值: {format_time(stats2_b['mean'])}")
    print(f"  标准差: {format_time(stats2_b['std'])}")
    print(f"  最小值: {format_time(stats2_b['min'])}")
    print(f"  最大值: {format_time(stats2_b['max'])}")

if __name__ == "__main__":
    file1 = "/home/panenbao/vllm/logs/schedule_and_execute_time.log"
    file2 = "/home/panenbao/results/schedule_and_execute_time.log"
    compare_files(file1, file2)