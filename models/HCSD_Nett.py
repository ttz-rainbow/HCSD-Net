import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath,trunc_normal_
from einops import rearrange,repeat

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x

class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class AFF_(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF_, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)

class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out

class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane-3, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LinearProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        attn_kv = x if attn_kv is None else attn_kv
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1]
        return q, k, v

def window_partition(x, win_size, dilation_rate=1,Hr_t=321, Wr_t=481):
    B, H, W, C = x.shape

    if dilation_rate != 1:
        up = nn.Upsample(size=(Hr_t,Wr_t),mode='nearest')
        x = x.permute(0,3,1,2)  #[1, 32, 321, 481]
        x = up(x)  #[1, 32, 328, 488]
        x = x.permute(0,2,3,1)  #[1, 328, 488, 32]
        B_r, H_r, W_r, C_r = x.shape
        x = x.view(B_r, H_r // win_size, win_size, W_r // win_size, win_size, C_r)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C_r)
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C)  # B' ,Wh ,Ww ,C

    return windows

def window_reverse(windows, win_size, H, W, dilation_rate=1,Hr_t=321, Wr_t=481):

    B = int(windows.shape[0] / (H * W / win_size / win_size))  #B = 1

    if dilation_rate != 1:
        x = windows.view(B, Hr_t // win_size, Wr_t // win_size, win_size, win_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hr_t, Wr_t, -1)
        x = x.permute(0,3,1,2)
        x = F.interpolate(x,size=(H,W),mode='nearest')
        x = x.permute(0, 2, 3, 1)
    else:
        x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)

    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, token_projection='linear', qkv_bias=True,attn_drop=0.,
                 proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.win_size = [8,8]  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.win_size[0] - 1) * (2 * self.win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        coords_h = torch.arange(self.win_size[0])  # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1])  # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1

        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)


        self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)

        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.se_layer = nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape  ##[2048, 64, 32]
        q, k, v = self.qkv(x, attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1) // relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d=ratio)

        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.se_layer(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}'

class WSA_LFE(nn.Module):
    def __init__(self,dim,num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,pool_size=2,):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.window_size = 8
        self.attn_xc = WindowAttention(dim, num_heads=num_heads,
                                               qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
    def forward(self,xc):

        B, H, W, C = xc.shape  #[8, 128, 128, 32]
        xc = self.norm1(xc)
        shifted_x = xc
        if H % self.window_size == 0 and W % self.window_size == 0:
            dilation_rate = 1
        else:
            dilation_rate = 2
        if dilation_rate != 1:
            H_rate = (H // self.window_size + 1) * self.window_size  # 328
            W_rate = (W // self.window_size + 1) * self.window_size  # 488
        else:
            H_rate = H
            W_rate = W
        x_windows = window_partition(shifted_x, self.window_size,dilation_rate=dilation_rate,Hr_t=H_rate,Wr_t=W_rate)  ## nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  ## nW*B, window_size*window_size, C
        attn_windows = self.attn_xc(x_windows)  # nW*B, window_size*window_size, C
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W,dilation_rate=dilation_rate,Hr_t=H_rate,Wr_t=W_rate)  # B H' W' C

        x = shifted_x   #[8, 128, 128, 32]
        x = x.permute(0,3,1,2)

        return x

class high_feature_extractor(nn.Module):
    def __init__(self,dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
        self.conv = nn.Conv2d(dim,dim*2,kernel_size=1,stride=1,padding=0,bias=False)
        self.pro = nn.Conv2d(dim*2,dim*2, kernel_size=1, stride=1, padding=0)
        self.gelu1 = nn.GELU()

    def forward(self,x):
        x = self.Maxpool(x)
        x = x.permute(0,3,1,2)
        x = self.conv(x)
        x = self.pro(x)
        x = self.gelu1(x)

        return x

class Mixer(nn.Module):
    def __init__(self,dim,num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,pool_size=2,):
        super().__init__()
        self.num_heads = num_heads
        dim_x = dim_xc = dim
        self.HFE = high_feature_extractor(dim_x, kernel_size=3, stride=1, padding=1)
        self.LFE = WSA_LFE(dim_xc, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, pool_size=pool_size)
        self.conv_fuse = nn.Conv2d(dim_xc + dim_x*2, dim_xc + dim_x*2, kernel_size=3, stride=1, padding=1,
                                   bias=False, groups=dim_xc + dim_x*2)
        self.proj = nn.Conv2d(dim_xc + dim_x*2, dim, kernel_size=1, stride=1, padding=0)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self,x,xc):
        x = self.HFE(x)  #[4, 64, 224, 224]
        xc = self.LFE(xc)   #[1, 32, 224, 224]

        x_fuse = torch.cat((x,xc),dim=1)  #[1, 96, 224, 224]
        x_out = x_fuse + self.conv_fuse(x_fuse)  #[1, 96, 224, 224]
        x_out = self.proj(x_out)   #[1, 32, 224, 224]
        x_out = self.proj_drop(x_out)   #[1, 32, 224, 224]
        x_out = x_out.permute(0,2,3,1).contiguous()   #[1, 224, 224, 32]

        return x_out

class ResDetail_Block(nn.Module):
    def __init__(self,dim,drop_path=0., mlp_ratio=4., act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm2 = nn.LayerNorm(dim)
    def forward(self,x_fuse):
        x_fuse = x_fuse + self.drop_path(x_fuse)
        x_fuse = x_fuse + self.drop_path(self.mlp(self.norm2(x_fuse)))
        return x_fuse

class Mformer_block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, pool_size=2,
                 ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.Mixer = Mixer(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                           proj_drop=0.,pool_size=pool_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.RDB = ResDetail_Block(dim, drop_path=drop_path, mlp_ratio=mlp_ratio)
    def forward(self,x,xc):
        x = self.norm1(x)
        xc = self.norm1(xc)
        x_fuse = self.Mixer(x,xc)
        x_fuse = self.RDB(x_fuse)

        return x_fuse

class HCSD_Net(nn.Module):
    def __init__(self,feats=32,num_res=8,base_channel=32):
        super(HCSD_Net, self).__init__()
        self.feats = feats
        self.conv = nn.Conv2d(3, self.feats, kernel_size=3, padding=1, stride=1)
        self.conv_c = nn.Conv2d(1, self.feats, kernel_size=3, padding=1, stride=1)

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.En_Mformer = nn.ModuleList([
            Mformer_block(base_channel, num_heads=8),
            Mformer_block(base_channel*2, num_heads=8),
        ])

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel * 2, num_res),
        ])

        self.AFFs_ = nn.ModuleList([
            AFF_(base_channel * 3, base_channel * 1),
            AFF_(base_channel * 3, base_channel * 2)
        ])
        self.FAM = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

        self.Convs = nn.ModuleList([
            BasicConv(base_channel, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 3, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.HFFMs = nn.ModuleList([
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

    def forward(self, x, xc):
        # xc = torch.randn([1, 1, 480, 640])
        x0 = self.conv(x)  #[4, 32, 224, 224]
        xc = self.conv_c(xc)   #[4, 32, 224, 224]

        ########### GSR Branch###########
        x0 = x0.permute(0,2,3,1)
        xc = xc.permute(0,2,3,1)
        res_x = self.En_Mformer[0](x0,xc)   #[1, 224, 224, 32]
        res_x = res_x.permute(0,3,1,2)
        res_x = self.Encoder[0](res_x)

        z = self.feat_extract[1](res_x)  #[1, 64, 112, 112]
        z = z.permute(0,2,3,1)

        t_z = z.permute(0,3,1,2)
        B0, C0, H0, W0 = t_z.shape
        # print('t2',t_z.shape)

        ########## MFI Branch###########
        x_2 = F.interpolate(x, size=(H0,W0))
        z2 = self.SCM2(x_2)  # [4, 64, 112, 112]
        z2_t = z2.permute(0, 2, 3, 1)

        z = self.En_Mformer[1](z,z2_t)   #[1, 112, 112, 64]
        z = z.permute(0,3,1,2)
        z = self.FAM(z,z2)
        res_x_2 = self.Encoder[1](z)

        ############# Multi-Scale Interaction#############
        B1, C1, H1, W1 = res_x.shape
        z12 = F.interpolate(res_x, size=(H0,W0))
        z21 = F.interpolate(res_x_2, size=(H1,W1))

        res_x = self.AFFs_[1](z21,res_x)   #[1, 64, 224, 224]
        res_x_2 = self.AFFs_[0](z12,res_x_2)   #[1, 32, 112, 112]

        outputs = list()
        z = self.Convs[0](res_x_2)  #[1, 64, 112, 112]
        z = self.HFFMs[0](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        z_tt = F.interpolate(z, size=(H1,W1))
        outputs.append(z_ + x_2)  # output[0]是x/2 的支路

        z = torch.cat([z_tt, res_x], dim=1)   #[1, 96, 224, 224]
        z = self.Convs[1](z)
        z = self.HFFMs[1](z)
        z = self.feat_extract[5](z)   #[1, 3, 224, 224]
        outputs.append(z + x)  # output[1]是原始分辨率的输出

        return outputs

# from ptflops import get_model_complexity_info
def model_complex(model, input_shape):
    macs, params = get_model_complexity_info(model, input_shape, as_strings=True,
                                             print_per_layer_stat=False, verbose=True)
    print(f'====> Number of Model Params: {params}')
    print(f'====> Computational complexity: {macs}')

if __name__=='__main__':
    model = HCSD_Net()
    model_complex(model, (3, 480, 640))     #model parameters与图像分辨率大小没有关系