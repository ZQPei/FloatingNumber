import numpy as np
import struct

def float_to_binary(number):
    return format(struct.unpack('!I', struct.pack('!f', number))[0], '032b')

def double_to_binary(number):
    return format(struct.unpack('!Q', struct.pack('!d', number))[0], '064b')

def float16_to_binary(float_number):
    return format(np.array(float_number, dtype=np.float16).view(np.uint16).item(), '016b')

def bfloat16_to_binary(float_number):
    # 将float32转换为字节
    float32_bytes = struct.pack('!f', float_number)
    # 将字节转换为int，然后转换为二进制字符串
    float32_bin = format(struct.unpack('!I', float32_bytes)[0], '032b')
    # 我们需要float32的符号位、指数位和bfloat16的尾数部分
    bf16_bin = float32_bin[:16]
    return bf16_bin

if __name__ == "__main__":
    # 示例浮点数
    float_number32 = 3.14  # float32
    float_number64 = 3.14  # float64
    float_number16 = np.float16(3.14)  # float16
    float_number_bf16 = 3.14  # bfloat16

    # 转换并输出结果
    print("Float32 to binary:", float_to_binary(float_number32))
    print("Float64 to binary:", double_to_binary(float_number64))
    print("Float16 to binary:", float16_to_binary(float_number16))
    print("Bfloat16 to binary:", bfloat16_to_binary(float_number_bf16))
