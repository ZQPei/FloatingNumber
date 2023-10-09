import numpy as np
import struct

def binary_to_float(binary_string):
    return struct.unpack('!f', struct.pack('!I', int(binary_string, 2)))[0]

def binary_to_double(binary_string):
    return struct.unpack('!d', struct.pack('!Q', int(binary_string, 2)))[0]

def binary_to_float16(binary_string):
    return np.array(int(binary_string, 2), dtype=np.uint16).view(np.float16).item()

def binary_to_bfloat16(binary_string):
    # 将二进制字符串转换为int
    int_number = int(binary_string, 2)
    # 将int转换为bytes
    bf16_bytes = int.to_bytes(int_number, length=2, byteorder='big', signed=False)
    # 将bytes转换为float32，然后转换为bfloat16
    float32_number = struct.unpack('!f', bf16_bytes + b'\x00\x00')[0]
    return float32_number

if __name__ == "__main__":
    # 示例二进制字符串
    binary_string32 = "01000000010010010000111111011011"  # 对应 float32
    binary_string64 = "0100000000001001001000011111101101010100010001000010110100011000"  # 对应 float64
    binary_string16 = "0100001001001000"  # 对应 float16
    binary_string_bf16 = "0100000001001001"  # 对应 bfloat16
    
    # 转换并输出结果
    print("Binary to float32:", binary_to_float(binary_string32))
    print("Binary to float64:", binary_to_double(binary_string64))
    print("Binary to float16:", binary_to_float16(binary_string16))
    print("Binary to bfloat16:", binary_to_bfloat16(binary_string_bf16))
