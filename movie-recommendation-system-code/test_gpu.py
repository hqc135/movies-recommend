#!/usr/bin/env python3
"""
GPU状态测试脚本
用于检查TensorFlow GPU支持状态
"""

import tensorflow as tf
import sys

def test_gpu_availability():
    """测试GPU可用性"""
    print("=" * 60)
    print("TensorFlow GPU 状态检查")
    print("=" * 60)
    
    # TensorFlow版本
    print(f"TensorFlow版本: {tf.__version__}")
    
    # 检查GPU设备
    physical_devices = tf.config.list_physical_devices('GPU')
    print(f"可用的GPU设备数量: {len(physical_devices)}")
    
    if len(physical_devices) > 0:
        print("\nGPU设备信息:")
        for i, device in enumerate(physical_devices):
            print(f"  GPU {i}: {device}")
            
        # 测试GPU内存增长设置
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print("\n✓ GPU内存增长设置成功")
        except Exception as e:
            print(f"\n✗ GPU内存增长设置失败: {e}")
            
        # 测试混合精度
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("✓ 混合精度策略设置成功")
        except Exception as e:
            print(f"✗ 混合精度策略设置失败: {e}")
            
        # 简单GPU计算测试
        try:
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
                c = tf.matmul(a, b)
                print(f"\n✓ GPU计算测试成功:")
                print(f"  矩阵乘法结果: \n{c.numpy()}")
        except Exception as e:
            print(f"\n✗ GPU计算测试失败: {e}")
            
        # 检查CUDA和cuDNN
        try:
            print(f"\nCUDA版本: {tf.test.is_built_with_cuda()}")
            print(f"cuDNN支持: {tf.test.is_built_with_gpu_support()}")
        except:
            print("\n无法获取CUDA/cuDNN信息")
            
    else:
        print("\n⚠️  未检测到GPU设备")
        print("可能的原因:")
        print("1. 系统中没有兼容的NVIDIA GPU")
        print("2. 没有安装CUDA驱动程序")
        print("3. 没有安装cuDNN库")
        print("4. TensorFlow版本与CUDA版本不匹配")
        print("\n将使用CPU进行计算...")
        
    print("\n" + "=" * 60)

def test_cpu_performance():
    """测试CPU性能作为对比"""
    print("CPU性能测试...")
    import time
    
    with tf.device('/CPU:0'):
        start_time = time.time()
        # 创建较大的矩阵进行计算
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
        cpu_time = time.time() - start_time
        print(f"CPU矩阵乘法 (1000x1000) 用时: {cpu_time:.4f} 秒")
    
    # 如果有GPU，测试GPU性能
    if len(tf.config.list_physical_devices('GPU')) > 0:
        with tf.device('/GPU:0'):
            start_time = time.time()
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)
            gpu_time = time.time() - start_time
            print(f"GPU矩阵乘法 (1000x1000) 用时: {gpu_time:.4f} 秒")
            
            if cpu_time > 0:
                speedup = cpu_time / gpu_time
                print(f"GPU加速比: {speedup:.2f}x")

if __name__ == "__main__":
    test_gpu_availability()
    test_cpu_performance()
