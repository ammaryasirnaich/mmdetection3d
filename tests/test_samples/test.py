import torch
# patch_score shape torch.Size([2, 512, 256])
# pos_score shape torch.Size([2, 512, 256])
# gating shape torch.Size([1, 4, 1, 1])

patch_score = torch.randn([2,512,256])
pos_score = torch.randn([2,512,256])
gating = torch.randn([1,2, 1, 1])


print("patch_score", patch_score.shape)
print("pos_score",pos_score.shape)
print("gating",gating.shape)

attn = (1.-torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score

print("attn", attn.shape)