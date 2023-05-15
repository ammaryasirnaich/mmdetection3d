import torch
import torch.nn as nn
import torch.nn.functional as F

# Optionally use the context manager to ensure one of the fused kerenels is run
query = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
key = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
value = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")

value_2 = torch.eye(128, 128, dtype=torch.float16, device="cuda")

value_3 = torch.eye(64, 64, dtype=torch.float16, device="cuda")

output_1 = F.scaled_dot_product_attention(query,key,value)
print("shape", output_1.shape, "type:", output_1.dtype)



# output_3 = F.scaled_dot_product_attention(query,key,value_2)
# output_3 = F.scaled_dot_product_attention(output_3,value.transpose(-2,-1),value_2)

# print("shape", output_3.shape, "type:", output_3.dtype)


# assert output_1 == output_3
# print("output_3 pass")





output_4 = F.scaled_dot_product_attention(query,key,value_2)
print("output_4 first" , output_4.shape)
output_4 = F.scaled_dot_product_attention(output_4,value.transpose(-2,-1),value_3)
print("output_4 second" , output_4.shape)

# print(torch.isclose(output_1, output_4))
print(output_1-output_4)

# assert output_1 == output_4
print("output_4 pass")




# pos_score = (query @ key.transpose(-2, -1))
# print("pos_score shape" , pos_score.shape)


# output_2 = pos_score.softmax(dim=-1) @ value

# print("shape", output_2.shape,"type:", output_2.dtype)


# print("output_1 result" ,output_1[0,0,0,:] )
# print("output_2 result" ,output_2[0,0,0,:] )

# diff = output_1 - output_2

# print("diff" , torch.abs(diff).sum(dim=0))



