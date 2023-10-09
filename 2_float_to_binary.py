import numpy as np
import torch
import struct

def float_to_binary(number):
    return format(struct.unpack('!I', struct.pack('!f', number))[0], '032b')

def double_to_binary(number):
    return format(struct.unpack('!Q', struct.pack('!d', number))[0], '064b')

def float16_to_binary(float_number):
    return format(np.array(float_number, dtype=np.float16).view(np.uint16).item(), '016b')

def bfloat16_to_binary(float_number):
    # 将bfloat16转换为float32，然后获取其二进制表示
    bf16_number = torch.tensor([float_number], dtype=torch.bfloat16)
    float32_number = bf16_number.to(dtype=torch.float32).numpy()
    float32_binary = np.unpackbits(np.frombuffer(float32_number.tobytes(), dtype=np.uint8))
    # 我们只需要float32的尾数部分和bfloat16的指数部分
    return ''.join(map(str, float32_binary[:9].tolist() + float32_binary[16:16+7].tolist()))

if __name__ == "__main__":
    # 示例浮点数
    float_number32 = 3.14  # float32
    float_number64 = 3.14  # float64
    float_number16 = np.float16(3.14)  # float16
    float_number_bf16 = 3.14  # bfloat16
    # 01000000010010001111010111000011
    # 转换并输出结果
    print("Float32 to binary:", float_to_binary(float_number32))
    print("Float64 to binary:", double_to_binary(float_number64))
    print("Float16 to binary:", float16_to_binary(float_number16))
    print("Bfloat16 to binary:", bfloat16_to_binary(float_number_bf16))
