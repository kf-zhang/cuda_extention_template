import torch

library_path = './build/lib.linux-x86_64-cpython-312/custom_ops.cpython-312-x86_64-linux-gnu.so'
torch.ops.load_library(library_path)

x = torch.ones(3).cuda()
y = torch.ones(3).cuda()
z = torch.ops.my_ops.elementwise_add(x, y)
print(z)