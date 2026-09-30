import torch.nn as nn
import torch
import torch.nn.functional as F

from escnn import gspaces
from escnn.nn import R2Conv, R2ConvTransposed, GeometricTensor, Linear, FieldType, ELU, R2Upsampling
import math


class CNNAutoencoder2D(nn.Module):
    def __init__(self,
                 dims=None, 
                 encoder_channels=None, 
                 decoder_channels=None,
                 encoder_fully_connected_layers_sizes=None, 
                 decoder_fully_connected_layers_sizes=None,
                 activation_function=nn.ELU(),
                 encoder_kernel_sizes=5,
                 encoder_paddings=2, 
                 encoder_strides=2,
                 decoder_kernel_sizes=5, 
                 decoder_paddings=2, 
                 decoder_strides=2):

        super(CNNAutoencoder2D, self).__init__()

        def _as_list(v, n):
            if isinstance(v, (list, tuple)):
                assert len(v) == n, f"Expected {n} values, got {len(v)}"
                return list(v)
            return [int(v)] * n
        
        self.number_of_convolutional_layers_encoder = len(encoder_channels)
        self.encoder_channels = encoder_channels
        self.number_of_fully_connected_layers_encoder = len(encoder_fully_connected_layers_sizes)
        self.encoder_fully_connected_layers_sizes = encoder_fully_connected_layers_sizes
        self.number_of_convolutional_layers_decoder = len(decoder_channels)
        self.decoder_channels = decoder_channels
        self.number_of_fully_connected_layers_decoder = len(decoder_fully_connected_layers_sizes)
        self.decoder_fully_connected_layers_sizes = decoder_fully_connected_layers_sizes

        self.activation_function = activation_function

        self.C = int(dims[0])
        self.Nx = int(dims[1])
        self.Ny = int(dims[2])

        self.encoder_kernel_sizes = _as_list(encoder_kernel_sizes, self.number_of_convolutional_layers_encoder)
        self.encoder_paddings = _as_list(encoder_paddings, self.number_of_convolutional_layers_encoder)
        self.encoder_strides = _as_list(encoder_strides, self.number_of_convolutional_layers_encoder)

        self.decoder_kernel_sizes = _as_list(decoder_kernel_sizes, self.number_of_convolutional_layers_decoder)
        self.decoder_paddings = _as_list(decoder_paddings, self.number_of_convolutional_layers_decoder)
        self.decoder_strides = _as_list(decoder_strides, self.number_of_convolutional_layers_decoder)

        # ===================== ENCODER ===================== #
        class Encoder(nn.Module):
            def __init__(self, outer):
                super(Encoder,self).__init__()
                self.conv_layers = nn.ModuleList()
                self.fc_layers = nn.ModuleList()

                activation_function = outer.activation_function
                self.H = outer.Ny
                self.W = outer.Nx
                C = outer.C

                # Convolutional layers
                for i in range(outer.number_of_convolutional_layers_encoder):
                    k = outer.encoder_kernel_sizes[i]
                    p = outer.encoder_paddings[i]
                    s = outer.encoder_strides[i]

                    if i == 0:
                        self.conv_layers.extend([nn.Conv2d(C, outer.encoder_channels[i], kernel_size=k, stride=s, padding=p, padding_mode='circular')])
                    else:
                        self.conv_layers.extend([nn.Conv2d(outer.encoder_channels[i-1], outer.encoder_channels[i], kernel_size=k, stride=s, padding=p, padding_mode='circular')])

                    self.conv_layers.extend([activation_function])
                    self.H = self.conv_out(self.H, kernel_size=k, padding=p, stride=s)
                    self.W = self.conv_out(self.W, kernel_size=k, padding=p, stride=s)

                C_enc = outer.encoder_channels[-1]
                outer._enc_feat_shape = (C_enc, self.H, self.W)

                # Compute the first fc size automatically
                self._full_fc_sizes = [C_enc * self.H * self.W] + list(outer.encoder_fully_connected_layers_sizes)

                for i in range(len(self._full_fc_sizes) - 1):
                    self.fc_layers.extend([nn.Linear(self._full_fc_sizes[i], self._full_fc_sizes[i+1])])
                    if i < len(self._full_fc_sizes) - 2:
                        self.fc_layers.extend([activation_function])

            def forward(self, x):
                for layer in self.conv_layers:
                    x = layer(x)

                x = x.view(x.shape[0], -1)
                
                for layer in self.fc_layers:
                    x = layer(x)

                return x
            
            def conv_out(self, n, kernel_size=5, padding=2, stride=2, dilation=1):
                return ((n + 2*padding - dilation*(kernel_size-1) - 1)//stride) + 1

        # ===================== DECODER ===================== #
        class Decoder(nn.Module):

            def __init__(self, outer):
                super(Decoder,self).__init__()
                self.conv_layers = nn.ModuleList()
                self.fc_layers = nn.ModuleList()

                activation_function = outer.activation_function
                C = outer.C
                self.decoder_channels = outer.decoder_channels

                C_enc, H_enc, W_enc = outer._enc_feat_shape
                self.start_cl = (H_enc, W_enc)
                assert outer.decoder_channels[0] == C_enc

                # compute size of last fc layer automatically
                self._full_fc_sizes = list(outer.decoder_fully_connected_layers_sizes) + [outer.decoder_channels[0] * H_enc * W_enc]

                # Fully connected layers
                for i in range(len(self._full_fc_sizes) - 1):
                    self.fc_layers.extend([nn.Linear(self._full_fc_sizes[i], self._full_fc_sizes[i+1])])
                    if i < len(self._full_fc_sizes) - 2:
                        self.fc_layers.extend([activation_function])

                # compute the values for output padding such that the output size is again Nx x Ny
                ops = self.plan_output_padding(
                    target=outer.Ny,
                    layers=outer.number_of_convolutional_layers_decoder,
                    start=H_enc,
                    ks=outer.decoder_kernel_sizes,
                    ps=outer.decoder_paddings,
                    ss=outer.decoder_strides,
                    ds=[1]*outer.number_of_convolutional_layers_decoder
                )

                self._output_padding_list = ops
  
                # Transposed convolutional layers
                for i in range(outer.number_of_convolutional_layers_decoder):
                    k = outer.decoder_kernel_sizes[i]
                    p = outer.decoder_paddings[i]
                    s = outer.decoder_strides[i]
                    op = self._output_padding_list[i]

                    if i == outer.number_of_convolutional_layers_decoder - 1:
                        self.conv_layers.extend([nn.ConvTranspose2d(decoder_channels[i], C, kernel_size=k, stride=s, padding=p, output_padding=op)])
                    else:
                        self.conv_layers.extend([nn.ConvTranspose2d(decoder_channels[i], decoder_channels[i+1], kernel_size=k, stride=s, padding=p, output_padding=op)])
                       
                    if i != outer.number_of_convolutional_layers_decoder - 1:
                        self.conv_layers.extend([activation_function])
  
            def forward(self, x):
                for layer in self.fc_layers:
                    x = layer(x)

                B, _ = x.shape
                x = x.view(B, self.decoder_channels[0], self.start_cl[0], self.start_cl[1])

                # conv layers
                for layer in self.conv_layers:
                    x = layer(x)

                return x
            
            def deconv_out(self, n, k=5, p=2, s=2, d=1, op=0):
                return (n - 1) * s - 2*p + d*(k-1) + op + 1
            
            def plan_output_padding(self, target, layers, start=1, ks=None, ps=None, ss=None, ds=None):
                    from collections import deque
                    if ds is None: ds = [1]*layers
                    dq = deque([(start, [])])
                    visited = set()
                    while dq:
                        n, ops = dq.popleft()
                        if len(ops) == layers:
                            if n == target:
                                return ops
                            continue
                        idx = len(ops)
                        key = (n, idx)
                        if key in visited: 
                            continue
                        visited.add(key)
                        k = ks[idx]; p = ps[idx]; s = ss[idx]; d = ds[idx]
                        for op in range(s):  # op < stride
                            n2 = self.deconv_out(n, k, p, s, d, op)
                            dq.append((n2, ops + [op]))
                    raise ValueError(f"No output_padding sequence found for target={target} with layers={layers}.")
  
        self.encoder = Encoder(self)
        self.decoder = Decoder(self)
        self.apply(self._init_kaiming_for_relu_family)

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded

    def print_parameters(self):
        print("=> Parameters of neural network:")
        print("Encoder:")
        print(f'Convolutional layers: {int(self.number_of_convolutional_layers_encoder)}')
        print("Decoder:")
        print(f'Convolutional layers: {int(self.number_of_convolutional_layers_decoder)}')
        print("Architecture:")
        print(self)

    def encode(self, x):
        return self.encoder(x)
  
    def decode(self, x):
        return self.decoder(x)


#####################################################################################################


class UpsamplingCNNAutoencoder2D(nn.Module):
    def __init__(self,
                 dims=None, 
                 encoder_channels=None, 
                 decoder_channels=None,
                 encoder_fully_connected_layers_sizes=None, 
                 decoder_fully_connected_layers_sizes=None,
                 activation_function=nn.ELU(),
                 encoder_kernel_sizes=5,
                 encoder_paddings=2, 
                 encoder_strides=2,
                 decoder_kernel_sizes=5, 
                 decoder_paddings=2, 
                 decoder_strides=2):

        super(UpsamplingCNNAutoencoder2D, self).__init__()

        def _as_list(v, n):
            if isinstance(v, (list, tuple)):
                assert len(v) == n, f"Expected {n} values, got {len(v)}"
                return list(v)
            return [int(v)] * n
        
        self.number_of_convolutional_layers_encoder = len(encoder_channels)
        self.encoder_channels = encoder_channels
        self.number_of_fully_connected_layers_encoder = len(encoder_fully_connected_layers_sizes)
        self.encoder_fully_connected_layers_sizes = encoder_fully_connected_layers_sizes
        self.number_of_convolutional_layers_decoder = len(decoder_channels)
        self.decoder_channels = decoder_channels
        self.number_of_fully_connected_layers_decoder = len(decoder_fully_connected_layers_sizes)
        self.decoder_fully_connected_layers_sizes = decoder_fully_connected_layers_sizes

        self.activation_function = activation_function

        self.C = int(dims[0])
        self.Nx = int(dims[1])
        self.Ny = int(dims[2])

        self.encoder_kernel_sizes = _as_list(encoder_kernel_sizes, self.number_of_convolutional_layers_encoder)
        self.encoder_paddings = _as_list(encoder_paddings, self.number_of_convolutional_layers_encoder)
        self.encoder_strides = _as_list(encoder_strides, self.number_of_convolutional_layers_encoder)

        self.decoder_kernel_sizes = _as_list(decoder_kernel_sizes, self.number_of_convolutional_layers_decoder)
        self.decoder_paddings = _as_list(decoder_paddings, self.number_of_convolutional_layers_decoder)
        self.decoder_strides = _as_list(decoder_strides, self.number_of_convolutional_layers_decoder)

        # ===================== ENCODER ===================== #
        class Encoder(nn.Module):
            def __init__(self, outer):
                super(Encoder,self).__init__()
                self.conv_layers = nn.ModuleList()
                self.fc_layers = nn.ModuleList()

                activation_function = outer.activation_function
                self.H = outer.Ny
                self.W = outer.Nx
                C = outer.C

                # Convolutional layers
                for i in range(outer.number_of_convolutional_layers_encoder):
                    k = outer.encoder_kernel_sizes[i]
                    p = outer.encoder_paddings[i]
                    s = outer.encoder_strides[i]

                    if i == 0:
                        self.conv_layers.extend([nn.Conv2d(C, outer.encoder_channels[i], kernel_size=k, stride=s, padding=p, padding_mode='circular')])
                    else:
                        self.conv_layers.extend([nn.Conv2d(outer.encoder_channels[i-1], outer.encoder_channels[i], kernel_size=k, stride=s, padding=p, padding_mode='circular')])

                    self.conv_layers.extend([activation_function])
                    self.H = self.conv_out(self.H, kernel_size=k, padding=p, stride=s)
                    self.W = self.conv_out(self.W, kernel_size=k, padding=p, stride=s)

                C_enc = outer.encoder_channels[-1]
                outer._enc_feat_shape = (C_enc, self.H, self.W)
                self.dec_first = nn.Conv2d(outer.encoder_channels[-1], outer.encoder_channels[-1], kernel_size=self.H, stride=1, padding=0, bias=True)
                self.dec_first_act = activation_function

                # Compute the first fc size automatically
                self._full_fc_sizes = [outer.encoder_channels[-1]] + list(outer.encoder_fully_connected_layers_sizes)

                for i in range(len(self._full_fc_sizes) - 1):
                    self.fc_layers.extend([nn.Linear(self._full_fc_sizes[i], self._full_fc_sizes[i+1])])
                    if i < len(self._full_fc_sizes) - 2:
                        self.fc_layers.extend([activation_function])

            def forward(self, x):
                for layer in self.conv_layers:
                    x = layer(x)

                x = self.dec_first_act(self.dec_first(x))
                x = x.view(x.shape[0], -1)
                
                for layer in self.fc_layers:
                    x = layer(x)

                return x
            
            def conv_out(self, n, kernel_size=5, padding=2, stride=2, dilation=1):
                return ((n + 2*padding - dilation*(kernel_size-1) - 1)//stride) + 1

        # ===================== DECODER ===================== #
        class Decoder(nn.Module):

            def __init__(self, outer):
                super(Decoder,self).__init__()
                self.conv_layers = nn.ModuleList()
                self.fc_layers = nn.ModuleList()

                activation_function = outer.activation_function
                C = outer.C
                self.decoder_channels = outer.decoder_channels

                C_enc, H_enc, W_enc = outer._enc_feat_shape
                self.start_cl = (H_enc, W_enc)
                assert outer.decoder_channels[0] == C_enc

                # compute size of last fc layer automatically
                self._full_fc_sizes = list(outer.decoder_fully_connected_layers_sizes) + [outer.decoder_channels[0]]

                # Fully connected layers
                for i in range(len(self._full_fc_sizes) - 1):
                    self.fc_layers.extend([nn.Linear(self._full_fc_sizes[i], self._full_fc_sizes[i+1])])
                    if i < len(self._full_fc_sizes) - 2:
                        self.fc_layers.extend([activation_function])

                self.dec_first = nn.ConvTranspose2d(outer.encoder_channels[-1], outer.encoder_channels[-1], kernel_size=H_enc, stride=1, padding=0, output_padding=0, bias=True)
                self.dec_first_act = activation_function
                
                # Transposed convolutional layers
                for i in range(outer.number_of_convolutional_layers_decoder):
                    k = outer.decoder_kernel_sizes[i]
                    p = outer.decoder_paddings[i]
                    s = outer.decoder_strides[i]
                    #op = self._output_padding_list[i]

                    if i == outer.number_of_convolutional_layers_decoder - 1:
                        self.conv_layers.extend([nn.Upsample(scale_factor=s)])
                        self.conv_layers.extend([nn.Conv2d(decoder_channels[i], C, kernel_size=k, padding=p, stride=1, padding_mode='circular', bias=True)])
                    else:
                        self.conv_layers.extend([nn.Upsample(scale_factor=s)])
                        self.conv_layers.extend([nn.Conv2d(decoder_channels[i], decoder_channels[i+1], kernel_size=k, padding=p, stride=1, padding_mode='circular', bias=True)])

                    if i != outer.number_of_convolutional_layers_decoder - 1:
                        self.conv_layers.extend([activation_function])
  
            def forward(self, x):
                for layer in self.fc_layers:
                    x = layer(x)

                B, _ = x.shape
                x = x.view(B, self.decoder_channels[0], 1, 1)
                x = self.dec_first_act(self.dec_first(x))

                # conv layers
                for layer in self.conv_layers:
                    x = layer(x)

                return x
             
        self.encoder = Encoder(self)
        self.decoder = Decoder(self)

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded

    def print_parameters(self):
        print("=> Parameters of neural network:")
        print("Encoder:")
        print(f'Convolutional layers: {int(self.number_of_convolutional_layers_encoder)}')
        print("Decoder:")
        print(f'Convolutional layers: {int(self.number_of_convolutional_layers_decoder)}')
        print("Architecture:")
        print(self)

    def encode(self, x):
        return self.encoder(x)
  
    def decode(self, x):
        return self.decoder(x)


###########################################################################################################

class RotationUpsamplingGCNNAutoencoder2D(nn.Module):
    def __init__(self,
                 dims=None, 
                 encoder_channels=None,
                 decoder_channels=None,
                 encoder_fully_connected_layers_sizes=None,
                 decoder_fully_connected_layers_sizes=None,
                 gspace=gspaces.rot2dOnR2(N=4),
                 activation_function=ELU,
                 encoder_kernel_sizes=5,
                 encoder_paddings=2, 
                 encoder_strides=2,
                 decoder_kernel_sizes=5, 
                 decoder_paddings=2, 
                 decoder_strides=2):

        super(RotationUpsamplingGCNNAutoencoder2D, self).__init__()

        def _as_list(v, n):
            if isinstance(v, (list, tuple)):
                assert len(v) == n, f"Expected {n} values, got {len(v)}"
                return list(v)
            return [int(v)] * n
        
        self.number_of_convolutional_layers_encoder = len(encoder_channels)
        self.encoder_channels = encoder_channels
        self.number_of_fully_connected_layers_encoder = len(encoder_fully_connected_layers_sizes)
        self.encoder_fully_connected_layers_sizes = encoder_fully_connected_layers_sizes
        self.number_of_convolutional_layers_decoder = len(decoder_channels)
        self.decoder_channels = decoder_channels
        self.number_of_fully_connected_layers_decoder = len(decoder_fully_connected_layers_sizes)
        self.decoder_fully_connected_layers_sizes = decoder_fully_connected_layers_sizes

        self.gspace = gspace
        self.activation_function = activation_function

        self.C = int(dims[0])
        self.Nx = int(dims[1])
        self.Ny = int(dims[2])

        self.encoder_kernel_sizes = _as_list(encoder_kernel_sizes, self.number_of_convolutional_layers_encoder)
        self.encoder_paddings = _as_list(encoder_paddings, self.number_of_convolutional_layers_encoder)
        self.encoder_strides = _as_list(encoder_strides, self.number_of_convolutional_layers_encoder)

        self.decoder_kernel_sizes = _as_list(decoder_kernel_sizes, self.number_of_convolutional_layers_decoder)
        self.decoder_paddings = _as_list(decoder_paddings, self.number_of_convolutional_layers_decoder)
        self.decoder_strides = _as_list(decoder_strides, self.number_of_convolutional_layers_decoder)

        # ===================== ENCODER ===================== #
        class Encoder(nn.Module):
            def __init__(self, outer):
                super(Encoder,self).__init__()
                self.conv_layers = nn.ModuleList()
                self.fc_layers = nn.ModuleList()

                self.gspace = outer.gspace
                self.activation_function = outer.activation_function
                self.H = outer.Ny
                self.W = outer.Nx
                self.C = outer.C

                # Convolutional layers (no pooling; keep equivariance)
                for i in range(outer.number_of_convolutional_layers_encoder):
                    k = outer.encoder_kernel_sizes[i]
                    p = outer.encoder_paddings[i]
                    s = outer.encoder_strides[i]

                    if i == 0:
                        in_type  = FieldType(self.gspace, self.C * [self.gspace.trivial_repr])
                        out_type = FieldType(self.gspace, encoder_channels[i] * [self.gspace.regular_repr])
                    else:
                        in_type  = FieldType(self.gspace, encoder_channels[i-1] * [self.gspace.regular_repr])
                        out_type = FieldType(self.gspace, encoder_channels[i] * [self.gspace.regular_repr])
                    
                    self.conv_layers.extend([R2Conv(in_type, out_type, kernel_size=k, padding=p, stride=s, padding_mode='circular', bias=True)])
                    self.conv_layers.extend([self.activation_function(out_type, inplace=False)])

                    # track spatial size
                    self.H = self.conv_out(self.H, kernel_size=k, padding=p, stride=s)
                    self.W = self.conv_out(self.W, kernel_size=k, padding=p, stride=s)

                # Global equivariant conv to 1x1 (canonical bottleneck)
                self.global_in_type = out_type
                self.global_out_type = FieldType(self.gspace, outer.encoder_channels[-1] * [self.gspace.regular_repr])

                self.dec_first = R2Conv(self.global_in_type, self.global_out_type, kernel_size=self.H, stride=1, padding=0, bias=True)
                self.dec_first_act = activation_function(self.dec_first.out_type, inplace=False)

                C_enc = outer.encoder_channels[-1]
                outer._enc_feat_shape = (C_enc, self.H, self.W)

                # 0D gspace FC stack to vector latent
                gspace0d = gspaces.no_base_space(self.gspace.fibergroup)
                self.gspace0d = gspace0d
                self._full_fc_sizes = [outer.encoder_channels[-1]] + list(outer.encoder_fully_connected_layers_sizes)

                for i in range(len(self._full_fc_sizes) - 1):
                    in_type0  = FieldType(gspace0d, self._full_fc_sizes[i] * [gspace0d.regular_repr])
                    out_type0 = FieldType(gspace0d, self._full_fc_sizes[i+1] * [gspace0d.regular_repr])
                    self.fc_layers.extend([Linear(in_type0, out_type0)])
                    if i < len(self._full_fc_sizes) - 2:
                        self.fc_layers.extend([self.activation_function(out_type0, inplace=False)])

            def forward(self, x):
                x = GeometricTensor(x, FieldType(self.gspace, self.C * [self.gspace.trivial_repr]))
                for layer in self.conv_layers:
                    x = layer(x)

                x = self.dec_first_act(self.dec_first(x))

                # 0D linear stack -> vector latent
                B, Ch, _, _ = x.tensor.shape
                x= GeometricTensor(x.tensor.view(B, Ch), FieldType(self.gspace0d, self._full_fc_sizes[0] * [self.gspace0d.regular_repr]))
                for layer in self.fc_layers:
                    x = layer(x)

                return x.tensor
            
            def conv_out(self, n, kernel_size=5, padding=2, stride=2, dilation=1):
                return ((n + 2*padding - dilation*(kernel_size-1) - 1)//stride) + 1

        # ===================== DECODER ===================== #
        class Decoder(nn.Module):

            def __init__(self, outer):
                super(Decoder,self).__init__()
                self.conv_layers = nn.ModuleList()
                self.fc_layers = nn.ModuleList()

                self.gspace = outer.gspace
                self.activation_function = outer.activation_function
                self.C = outer.C
                self.encoder_channels = outer.encoder_channels

                C_enc, H_enc, W_enc = outer._enc_feat_shape
                self.start_cl = (H_enc, W_enc)
                assert outer.decoder_channels[0] == C_enc

                # 0D FC layers: from latent back to multiplicity of global_out_type
                gspace0d = gspaces.no_base_space(self.gspace.fibergroup)
                self.gspace0d = gspace0d

                # decoder_fully_connected_layers_sizes are hidden 0D sizes.
                # Last FC maps to the same size used by encoder's global_out_type.
                self._full_fc_sizes = list(outer.decoder_fully_connected_layers_sizes) + [outer.encoder_channels[-1]]

                for i in range(len(self._full_fc_sizes) - 1):
                    in_type0  = FieldType(gspace0d, self._full_fc_sizes[i] * [gspace0d.regular_repr])
                    out_type0 = FieldType(gspace0d, self._full_fc_sizes[i+1] * [gspace0d.regular_repr])
                    self.fc_layers.extend([Linear(in_type0, out_type0)])
                    if i < len(self._full_fc_sizes) - 2:
                        self.fc_layers.extend([self.activation_function(out_type0, inplace=False)])

                self.enc_out_type = FieldType(self.gspace, outer.encoder_channels[-1] * [self.gspace.regular_repr])
                self.dec_first = R2ConvTransposed(self.enc_out_type,
                                            FieldType(self.gspace, self.encoder_channels[-1] * [self.gspace.regular_repr]),
                                            kernel_size=H_enc,
                                            stride=1,
                                            padding=0, 
                                            output_padding=0, 
                                            bias=True)
                
                self.dec_first_act = self.activation_function(self.dec_first.out_type, inplace=False)
                
                # Mirror upsampling: use user-provided decoder lists
                # We start from the same type as encoder's last feature BEFORE global conv
                in_t = FieldType(self.gspace, outer.encoder_channels[-1] * [self.gspace.regular_repr])
                for i in range(outer.number_of_convolutional_layers_decoder):
                    k = outer.decoder_kernel_sizes[i]
                    p = outer.decoder_paddings[i]
                    s = outer.decoder_strides[i]

                    if i == outer.number_of_convolutional_layers_decoder - 1:
                        out_t = FieldType(self.gspace, self.C * [self.gspace.trivial_repr])
                    else:
                        out_t = FieldType(self.gspace, outer.decoder_channels[i+1] * [self.gspace.regular_repr])

                    self.conv_layers.extend([R2Upsampling(in_t, scale_factor=s)])
                    self.conv_layers.extend([R2Conv(in_t, out_t, kernel_size=k, padding=p, stride=1, padding_mode='circular', bias=True)])

                    if i != outer.number_of_convolutional_layers_decoder - 1:
                        self.conv_layers.extend([self.activation_function(out_t, inplace=False)])
                    in_t = out_t
  

            def forward(self, x):
                # 0D linear stack back to channels
                x = GeometricTensor(x, FieldType(self.gspace0d, self._full_fc_sizes[0] * [self.gspace0d.regular_repr]))
                for layer in self.fc_layers:
                    x = layer(x)

                B, Ch = x.tensor.shape
                x = GeometricTensor(x.tensor.view(B, Ch, 1, 1), FieldType(self.gspace, self._full_fc_sizes[-1] * [self.gspace.regular_repr]))

                # invert global conv to (H,W) canonical map
                x = self.dec_first_act(self.dec_first(x))

                # upsample to full size
                for i, layer in enumerate(self.conv_layers):
                    x = layer(x)

                return x.tensor

        self.encoder = Encoder(self)
        self.decoder = Decoder(self)
        #self.float()

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded

    def print_parameters(self):
        print("=> Parameters of neural network:")
        print("Encoder:")
        print(f'Convolutional layers: {int(self.number_of_convolutional_layers_encoder)}')
        print("Decoder:")
        print(f'Convolutional layers: {int(self.number_of_convolutional_layers_decoder)}')
        print("Architecture:")
        print(self)

    def encode(self, x):
        return self.encoder(x)
  
    def decode(self, x):
        return self.decoder(x)
    
    def export(self):
        """
        Export the trained equivariant autoencoder to a pure PyTorch model.
        This removes escnn dependencies and improves inference speed.
        
        Returns:
            nn.Module: A pure PyTorch autoencoder with the same functionality
        """        
        # Set model to eval mode (required for export)
        self.eval()
        
        class ExportedEncoder(nn.Module):
            def __init__(self, encoder):
                super(ExportedEncoder, self).__init__()
                self.layers = nn.ModuleList()
                
                # Export convolutional layers
                for i, layer in enumerate(encoder.conv_layers):
                    try:
                        self.layers.append(layer.export())
                    except NotImplementedError:
                        print(f"Warning: Encoder conv layer {i} ({type(layer).__name__}) does not support export. Keeping original.")
                        self.layers.append(layer)
                  
                # Export global conv
                self.dec_first = encoder.dec_first.export()
                self.dec_first_act = encoder.dec_first_act.export()
                
                # Export FC layers
                self.fc_layers = nn.ModuleList()
                for i, layer in enumerate(encoder.fc_layers):
                    try:
                        self.fc_layers.append(layer.export())
                    except NotImplementedError:
                        print(f"Warning: Encoder FC layer {i} ({type(layer).__name__}) does not support export. Keeping original.")
                        self.fc_layers.append(layer)
                    
            
            def forward(self, x):
                # Apply conv layers
                for layer in self.layers:
                    x = layer(x)
  
                # Apply global conv
                x = self.dec_first_act(self.dec_first(x))
  
                # Flatten and apply FC layers
                B, Ch, _, _ = x.shape
                x = x.view(B, Ch)
                
                for layer in self.fc_layers:
                    x = layer(x)
 
                return x
        
        class ExportedDecoder(nn.Module):
            def __init__(self, decoder):
                super(ExportedDecoder, self).__init__()
                self._full_fc_sizes = decoder._full_fc_sizes
                self.start_cl = decoder.start_cl
                
                # Export FC layers
                self.fc_layers = nn.ModuleList()
                for i, layer in enumerate(decoder.fc_layers):
                    if hasattr(layer, 'export'):
                        try:
                            self.fc_layers.append(layer.export())
                        except NotImplementedError:
                            print(f"Warning: Decoder FC layer {i} ({type(layer).__name__}) does not support export. Keeping original.")
                            self.fc_layers.append(layer)
                    else:
                        self.fc_layers.append(layer)
                
                # Export global deconv
                if hasattr(decoder.dec_first, 'export'):
                    self.dec_first = decoder.dec_first.export()
                else:
                    print(f"Warning: dec_first does not support export. Keeping original.")
                    self.dec_first = decoder.dec_first
                
                # Keep activation
                self.dec_first_act = decoder.dec_first_act.export()
                
                # Export conv layers
                self.conv_layers = nn.ModuleList()
                for i, layer in enumerate(decoder.conv_layers):
                    if hasattr(layer, 'export'):
                        try:
                            self.conv_layers.append(layer.export())
                        except NotImplementedError:
                            print(f"Warning: Decoder conv layer {i} ({type(layer).__name__}) does not support export. Keeping original.")
                            self.conv_layers.append(layer)
                    else:
                        print("BIN ICH JEMALS HIER")
                        # R2Upsampling and activations don't have export
                        self.conv_layers.append(layer)
            
            def forward(self, x):
                # Apply FC
                for layer in self.fc_layers:
                    x = layer(x)
                
                # Reshape to spatial
                B, Ch = x.shape
                x = x.view(B, Ch, 1, 1)
                
                # Apply global deconv
                x = self.dec_first_act(self.dec_first(x))
                
                # Apply upsampling conv layers
                for layer in self.conv_layers:
                    x = layer(x)
                
                return x
        
        class ExportedAutoencoder(nn.Module):
            def __init__(self, encoder, decoder):
                super(ExportedAutoencoder, self).__init__()
                self.encoder = encoder
                self.decoder = decoder
            
            def forward(self, x):
                return self.decoder(self.encoder(x))
            
            def encode(self, x):
                return self.encoder(x)
            
            def decode(self, x):
                return self.decoder(x)
        
        # Create exported encoder and decoder
        exported_encoder = ExportedEncoder(self.encoder)
        exported_decoder = ExportedDecoder(self.decoder)
        
        # Create exported autoencoder
        exported_model = ExportedAutoencoder(exported_encoder, exported_decoder)
        exported_model.eval()
        exported_model.float()

        return exported_model



############################################################################


class TrivialUpsamplingGCNNAutoencoder2D(nn.Module):
    def __init__(self,
                 dims=None, 
                 encoder_channels=None,
                 decoder_channels=None,
                 encoder_fully_connected_layers_sizes=None,
                 decoder_fully_connected_layers_sizes=None,
                 gspace=gspaces.rot2dOnR2(N=4),
                 activation_function=ELU,
                 encoder_kernel_sizes=5,
                 encoder_paddings=2, 
                 encoder_strides=2,
                 decoder_kernel_sizes=5, 
                 decoder_paddings=2, 
                 decoder_strides=2):

        super(TrivialUpsamplingGCNNAutoencoder2D, self).__init__()

        def _as_list(v, n):
            if isinstance(v, (list, tuple)):
                assert len(v) == n, f"Expected {n} values, got {len(v)}"
                return list(v)
            return [int(v)] * n
        
        self.number_of_convolutional_layers_encoder = len(encoder_channels)
        self.encoder_channels = encoder_channels
        self.number_of_fully_connected_layers_encoder = len(encoder_fully_connected_layers_sizes)
        self.encoder_fully_connected_layers_sizes = encoder_fully_connected_layers_sizes
        self.number_of_convolutional_layers_decoder = len(decoder_channels)
        self.decoder_channels = decoder_channels
        self.number_of_fully_connected_layers_decoder = len(decoder_fully_connected_layers_sizes)
        self.decoder_fully_connected_layers_sizes = decoder_fully_connected_layers_sizes

        self.gspace = gspace
        self.activation_function = activation_function

        self.C = int(dims[0])
        self.Nx = int(dims[1])
        self.Ny = int(dims[2])

        self.encoder_kernel_sizes = _as_list(encoder_kernel_sizes, self.number_of_convolutional_layers_encoder)
        self.encoder_paddings = _as_list(encoder_paddings, self.number_of_convolutional_layers_encoder)
        self.encoder_strides = _as_list(encoder_strides, self.number_of_convolutional_layers_encoder)

        self.decoder_kernel_sizes = _as_list(decoder_kernel_sizes, self.number_of_convolutional_layers_decoder)
        self.decoder_paddings = _as_list(decoder_paddings, self.number_of_convolutional_layers_decoder)
        self.decoder_strides = _as_list(decoder_strides, self.number_of_convolutional_layers_decoder)

        # ===================== ENCODER ===================== #
        class Encoder(nn.Module):
            def __init__(self, outer):
                super(Encoder,self).__init__()
                self.conv_layers = nn.ModuleList()
                self.fc_layers = nn.ModuleList()

                self.gspace = outer.gspace
                self.activation_function = outer.activation_function
                self.H = outer.Ny
                self.W = outer.Nx
                self.C = outer.C

                # Convolutional layers (no pooling; keep equivariance)
                for i in range(outer.number_of_convolutional_layers_encoder):
                    k = outer.encoder_kernel_sizes[i]
                    p = outer.encoder_paddings[i]
                    s = outer.encoder_strides[i]

                    if i == 0:
                        in_type  = FieldType(self.gspace, self.C * [self.gspace.trivial_repr])
                        out_type = FieldType(self.gspace, encoder_channels[i] * [self.gspace.trivial_repr])
                    else:
                        in_type  = FieldType(self.gspace, encoder_channels[i-1] * [self.gspace.trivial_repr])
                        out_type = FieldType(self.gspace, encoder_channels[i] * [self.gspace.trivial_repr])
                    
                    self.conv_layers.extend([R2Conv(in_type, out_type, kernel_size=k, padding=p, stride=s, padding_mode='circular', bias=True)])
                    self.conv_layers.extend([self.activation_function(out_type, inplace=False)])

                    # track spatial size
                    self.H = self.conv_out(self.H, kernel_size=k, padding=p, stride=s)
                    self.W = self.conv_out(self.W, kernel_size=k, padding=p, stride=s)

                # Global equivariant conv to 1x1 (canonical bottleneck)
                self.global_in_type = out_type
                self.global_out_type = FieldType(self.gspace, outer.encoder_channels[-1] * [self.gspace.trivial_repr])

                self.dec_first = R2Conv(self.global_in_type, self.global_out_type, kernel_size=self.H, stride=1, padding=0, bias=True)
                self.dec_first_act = activation_function(self.dec_first.out_type, inplace=False)

                C_enc = outer.encoder_channels[-1]
                outer._enc_feat_shape = (C_enc, self.H, self.W)

                # 0D gspace FC stack to vector latent
                gspace0d = gspaces.no_base_space(self.gspace.fibergroup)
                self.gspace0d = gspace0d
                self._full_fc_sizes = [outer.encoder_channels[-1]] + list(outer.encoder_fully_connected_layers_sizes)

                for i in range(len(self._full_fc_sizes) - 1):
                    in_type0  = FieldType(gspace0d, self._full_fc_sizes[i] * [gspace0d.trivial_repr])
                    out_type0 = FieldType(gspace0d, self._full_fc_sizes[i+1] * [gspace0d.trivial_repr])
                    self.fc_layers.extend([Linear(in_type0, out_type0)])
                    if i < len(self._full_fc_sizes) - 2:
                        self.fc_layers.extend([self.activation_function(out_type0, inplace=False)])

            def forward(self, x):
                x = GeometricTensor(x, FieldType(self.gspace, self.C * [self.gspace.trivial_repr]))
                for layer in self.conv_layers:
                    x = layer(x)

                x = self.dec_first_act(self.dec_first(x))

                # 0D linear stack -> vector latent
                B, Ch, _, _ = x.tensor.shape
                x= GeometricTensor(x.tensor.view(B, Ch), FieldType(self.gspace0d, self._full_fc_sizes[0] * [self.gspace0d.trivial_repr]))
                for layer in self.fc_layers:
                    x = layer(x)

                return x.tensor
            
            def conv_out(self, n, kernel_size=5, padding=2, stride=2, dilation=1):
                return ((n + 2*padding - dilation*(kernel_size-1) - 1)//stride) + 1

        # ===================== DECODER ===================== #
        class Decoder(nn.Module):

            def __init__(self, outer):
                super(Decoder,self).__init__()
                self.conv_layers = nn.ModuleList()
                self.fc_layers = nn.ModuleList()

                self.gspace = outer.gspace
                self.activation_function = outer.activation_function
                self.C = outer.C
                self.encoder_channels = outer.encoder_channels

                C_enc, H_enc, W_enc = outer._enc_feat_shape
                self.start_cl = (H_enc, W_enc)
                assert outer.decoder_channels[0] == C_enc

                # 0D FC layers: from latent back to multiplicity of global_out_type
                gspace0d = gspaces.no_base_space(self.gspace.fibergroup)
                self.gspace0d = gspace0d

                # decoder_fully_connected_layers_sizes are hidden 0D sizes.
                # Last FC maps to the same size used by encoder's global_out_type.
                self._full_fc_sizes = list(outer.decoder_fully_connected_layers_sizes) + [outer.encoder_channels[-1]]

                for i in range(len(self._full_fc_sizes) - 1):
                    in_type0  = FieldType(gspace0d, self._full_fc_sizes[i] * [gspace0d.trivial_repr])
                    out_type0 = FieldType(gspace0d, self._full_fc_sizes[i+1] * [gspace0d.trivial_repr])
                    self.fc_layers.extend([Linear(in_type0, out_type0)])
                    if i < len(self._full_fc_sizes) - 2:
                        self.fc_layers.extend([self.activation_function(out_type0, inplace=False)])

                self.enc_out_type = FieldType(self.gspace, outer.encoder_channels[-1] * [self.gspace.trivial_repr])
                self.dec_first = R2ConvTransposed(self.enc_out_type,
                                            FieldType(self.gspace, self.encoder_channels[-1] * [self.gspace.trivial_repr]),
                                            kernel_size=H_enc,
                                            stride=1,
                                            padding=0, 
                                            output_padding=0, 
                                            bias=True)
                
                self.dec_first_act = self.activation_function(self.dec_first.out_type, inplace=False)
                
                # Mirror upsampling: use user-provided decoder lists
                # We start from the same type as encoder's last feature BEFORE global conv
                in_t = FieldType(self.gspace, outer.encoder_channels[-1] * [self.gspace.trivial_repr])
                for i in range(outer.number_of_convolutional_layers_decoder):
                    k = outer.decoder_kernel_sizes[i]
                    p = outer.decoder_paddings[i]
                    s = outer.decoder_strides[i]

                    if i == outer.number_of_convolutional_layers_decoder - 1:
                        out_t = FieldType(self.gspace, self.C * [self.gspace.trivial_repr])
                    else:
                        out_t = FieldType(self.gspace, outer.decoder_channels[i+1] * [self.gspace.trivial_repr])

                    self.conv_layers.extend([R2Upsampling(in_t, scale_factor=s)])
                    self.conv_layers.extend([R2Conv(in_t, out_t, kernel_size=k, padding=p, stride=1, padding_mode='circular', bias=True)])

                    if i != outer.number_of_convolutional_layers_decoder - 1:
                        self.conv_layers.extend([self.activation_function(out_t, inplace=False)])
                    in_t = out_t
  

            def forward(self, x):
                # 0D linear stack back to channels
                x = GeometricTensor(x, FieldType(self.gspace0d, self._full_fc_sizes[0] * [self.gspace0d.trivial_repr]))
                for layer in self.fc_layers:
                    x = layer(x)

                B, Ch = x.tensor.shape
                x = GeometricTensor(x.tensor.view(B, Ch, 1, 1), FieldType(self.gspace, self._full_fc_sizes[-1] * [self.gspace.trivial_repr]))

                # invert global conv to (H,W) canonical map
                x = self.dec_first_act(self.dec_first(x))

                # upsample to full size
                for layer in self.conv_layers:
                    x = layer(x)

                return x.tensor

        self.encoder = Encoder(self)
        self.decoder = Decoder(self)
        #self.float()

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded

    def print_parameters(self):
        print("=> Parameters of neural network:")
        print("Encoder:")
        print(f'Convolutional layers: {int(self.number_of_convolutional_layers_encoder)}')
        print("Decoder:")
        print(f'Convolutional layers: {int(self.number_of_convolutional_layers_decoder)}')
        print("Architecture:")
        print(self)

    def encode(self, x):
        return self.encoder(x)
  
    def decode(self, x):
        return self.decoder(x)
    

#############################################################################




def _as_list(v, n):
    if isinstance(v, (list, tuple)):
        assert len(v) == n, f"Expected {n} values, got {len(v)}"
        return list(v)
    return [int(v)] * n


def _rot90_kernel(w, k):
    k = k % 4
    if k == 0:
        return w
    return torch.rot90(w, k, dims=(-2, -1))


def _circular_pad2d(x, padding):
    if padding == 0:
        return x
    return F.pad(x, (padding, padding, padding, padding), mode="circular")


def _make_activation(activation_function):
    if isinstance(activation_function, nn.Module):
        return activation_function
    if isinstance(activation_function, type) and issubclass(activation_function, nn.Module):
        return activation_function()
    if callable(activation_function):
        try:
            act = activation_function()
            if isinstance(act, nn.Module):
                return act
        except Exception:
            pass
    return nn.ELU()


class C4Conv2d_TrivialToRegular(nn.Module):
    def __init__(self, cin, cout_mult, kernel_size, stride=1, padding=0, bias=True, dilation=1):
        super().__init__()
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation = int(dilation)
        k = int(kernel_size)

        self.weight = nn.Parameter(torch.empty(int(cout_mult), int(cin), k, k))
        self.bias = nn.Parameter(torch.zeros(int(cout_mult))) if bias else None
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x):
        x = _circular_pad2d(x, self.padding)
        ys = []
        for r in range(4):
            w_r = _rot90_kernel(self.weight, r)
            y_r = F.conv2d(x, w_r, bias=self.bias, stride=self.stride, padding=0, dilation=self.dilation)
            ys.append(y_r)
        return torch.cat(ys, dim=1)


class C4Conv2d_RegularToRegular(nn.Module):
    def __init__(self, cin_mult, cout_mult, kernel_size, stride=1, padding=0, bias=True, dilation=1):
        super().__init__()
        self.cin_mult = int(cin_mult)
        self.cout_mult = int(cout_mult)
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation = int(dilation)
        k = int(kernel_size)

        self.weight = nn.Parameter(torch.empty(4, self.cout_mult, self.cin_mult, k, k))
        self.bias = nn.Parameter(torch.zeros(self.cout_mult)) if bias else None
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.cin_mult * 4, f"Expected {self.cin_mult*4} channels, got {C}"

        x = _circular_pad2d(x, self.padding)
        x_orient = [x[:, o*self.cin_mult:(o+1)*self.cin_mult] for o in range(4)]

        ys = []
        for r in range(4):
            y_r = None
            for s in range(4):
                delta = (s - r) % 4
                w = self.weight[delta]        
                w_rs = _rot90_kernel(w, r)     
                contrib = F.conv2d(
                    x_orient[s], w_rs,
                    bias=None,
                    stride=self.stride,
                    padding=0,
                    dilation=self.dilation
                )
                y_r = contrib if y_r is None else (y_r + contrib)

            if self.bias is not None:
                y_r = y_r + self.bias.view(1, -1, 1, 1)

            ys.append(y_r)

        return torch.cat(ys, dim=1)


class C4Conv2d_RegularToTrivial(nn.Module):
    def __init__(self, cin_mult, cout, kernel_size, stride=1, padding=0, bias=True, dilation=1):
        super().__init__()
        self.cin_mult = int(cin_mult)
        self.cout = int(cout)
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation = int(dilation)
        k = int(kernel_size)

        self.weight = nn.Parameter(torch.empty(self.cout, self.cin_mult, k, k))
        self.bias = nn.Parameter(torch.zeros(self.cout)) if bias else None
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.cin_mult * 4, f"Expected {self.cin_mult*4} channels, got {C}"

        x = _circular_pad2d(x, self.padding)
        x_orient = [x[:, o*self.cin_mult:(o+1)*self.cin_mult] for o in range(4)]

        y = None
        for r in range(4):
            w_r = _rot90_kernel(self.weight, r)
            contrib = F.conv2d(
                x_orient[r], w_r,
                bias=None,
                stride=self.stride,
                padding=0,
                dilation=self.dilation
            )
            y = contrib if y is None else (y + contrib)

        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)
        return y


class C4ConvTranspose2d_RegularToRegular(nn.Module):
    def __init__(self, cin_mult, cout_mult, kernel_size, stride=1, padding=0, output_padding=0, bias=True, dilation=1):
        super().__init__()
        self.cin_mult = int(cin_mult)
        self.cout_mult = int(cout_mult)
        self.stride = int(stride)
        self.padding = int(padding)
        self.output_padding = int(output_padding)
        self.dilation = int(dilation)
        k = int(kernel_size)

        self.weight = nn.Parameter(torch.empty(4, self.cout_mult, self.cin_mult, k, k))
        self.bias = nn.Parameter(torch.zeros(self.cout_mult)) if bias else None
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.cin_mult * 4, f"Expected {self.cin_mult*4} channels, got {C}"

        x_orient = [x[:, o*self.cin_mult:(o+1)*self.cin_mult] for o in range(4)]

        ys = []
        for r in range(4):
            y_r = None
            for s in range(4):
                delta = (s - r) % 4
                w = self.weight[delta]             
                w_t = w.permute(1, 0, 2, 3)         
                w_rs = _rot90_kernel(w_t, r)
                contrib = F.conv_transpose2d(
                    x_orient[s], w_rs,
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                    output_padding=self.output_padding,
                    dilation=self.dilation
                )
                y_r = contrib if y_r is None else (y_r + contrib)

            if self.bias is not None:
                y_r = y_r + self.bias.view(1, -1, 1, 1)

            ys.append(y_r)

        return torch.cat(ys, dim=1)


class C4LinearRegularToRegular(nn.Module):
    def __init__(self, in_mult, out_mult, bias=True):
        super().__init__()
        self.in_mult = int(in_mult)
        self.out_mult = int(out_mult)
        self.weight = nn.Parameter(torch.empty(4, self.out_mult, self.in_mult))
        self.bias = nn.Parameter(torch.zeros(self.out_mult)) if bias else None
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x):
        B, D = x.shape
        assert D == self.in_mult * 4, f"Expected {self.in_mult*4} features, got {D}"
        xs = [x[:, o*self.in_mult:(o+1)*self.in_mult] for o in range(4)]

        ys = []
        for r in range(4):
            y_r = None
            for s in range(4):
                delta = (s - r) % 4
                W = self.weight[delta]
                contrib = xs[s] @ W.t()
                y_r = contrib if y_r is None else (y_r + contrib)
            if self.bias is not None:
                y_r = y_r + self.bias.view(1, -1)
            ys.append(y_r)

        return torch.cat(ys, dim=1)


class C4Upsample(nn.Module):
    def __init__(self, scale_factor, mode="nearest"):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode=mode)

    def forward(self, x):
        return self.up(x)


class RotationUpsamplingGCNN2D_TorchOnly(nn.Module):
    """
    Torch-only analogue of RotationUpsamplingGCNNAutoencoder2D (escnn),
    but implemented with explicit C4 orientation channels and tied/rotated weights.
    """

    def __init__(self,
                 dims=None,
                 encoder_channels=None,
                 decoder_channels=None,
                 encoder_fully_connected_layers_sizes=None,
                 decoder_fully_connected_layers_sizes=None,
                 gspace=None, 
                 activation_function=nn.ELU(),
                 encoder_kernel_sizes=5,
                 encoder_paddings=2,
                 encoder_strides=2,
                 decoder_kernel_sizes=5,
                 decoder_paddings=2,
                 decoder_strides=2):
        super().__init__()

        assert dims is not None and len(dims) == 3, "dims must be (C, Nx, Ny)"
        self.C = int(dims[0])
        self.Nx = int(dims[1])
        self.Ny = int(dims[2])

        self.encoder_channels = list(encoder_channels)
        self.decoder_channels = list(decoder_channels)
        self.encoder_fully_connected_layers_sizes = list(encoder_fully_connected_layers_sizes)
        self.decoder_fully_connected_layers_sizes = list(decoder_fully_connected_layers_sizes)

        self.number_of_convolutional_layers_encoder = len(self.encoder_channels)
        self.number_of_convolutional_layers_decoder = len(self.decoder_channels)

        self.encoder_kernel_sizes = _as_list(encoder_kernel_sizes, self.number_of_convolutional_layers_encoder)
        self.encoder_paddings    = _as_list(encoder_paddings,    self.number_of_convolutional_layers_encoder)
        self.encoder_strides     = _as_list(encoder_strides,     self.number_of_convolutional_layers_encoder)

        self.decoder_kernel_sizes = _as_list(decoder_kernel_sizes, self.number_of_convolutional_layers_decoder)
        self.decoder_paddings    = _as_list(decoder_paddings,     self.number_of_convolutional_layers_decoder)
        self.decoder_strides     = _as_list(decoder_strides,      self.number_of_convolutional_layers_decoder)

        self.activation_function = _make_activation(activation_function)

        # ===================== ENCODER ===================== #
        class Encoder(nn.Module):
            def __init__(self, outer):
                super().__init__()
                self.conv_layers = nn.ModuleList()
                self.fc_layers = nn.ModuleList()

                act = outer.activation_function
                H = outer.Ny
                W = outer.Nx

                # Conv stack: trivial -> regular -> regular -> ...
                for i in range(outer.number_of_convolutional_layers_encoder):
                    k = outer.encoder_kernel_sizes[i]
                    p = outer.encoder_paddings[i]
                    s = outer.encoder_strides[i]

                    if i == 0:
                        self.conv_layers.append(
                            C4Conv2d_TrivialToRegular(outer.C, outer.encoder_channels[i],
                                                     kernel_size=k, stride=s, padding=p, bias=True)
                        )
                    else:
                        self.conv_layers.append(
                            C4Conv2d_RegularToRegular(outer.encoder_channels[i-1], outer.encoder_channels[i],
                                                     kernel_size=k, stride=s, padding=p, bias=True)
                        )
                    self.conv_layers.append(act)

                    # update spatial size (standard conv formula)
                    H = ((H + 2*p - (k-1) - 1) // s) + 1
                    W = ((W + 2*p - (k-1) - 1) // s) + 1

                # store encoder feature shape on outer (like your code does)
                C_enc_mult = outer.encoder_channels[-1]
                outer._enc_feat_shape = (C_enc_mult, H, W)

                # global conv to 1x1 (equivariant): regular -> regular
                self.enc_global = C4Conv2d_RegularToRegular(
                    cin_mult=C_enc_mult,
                    cout_mult=C_enc_mult,
                    kernel_size=H,   # assuming square in your setup; matches your style
                    stride=1,
                    padding=0,
                    bias=True
                )
                self.enc_global_act = act

                # 0D FC sizes are multiplicities (regular repr), not raw dims
                self._full_fc_mults = [C_enc_mult] + list(outer.encoder_fully_connected_layers_sizes)

                for i in range(len(self._full_fc_mults) - 1):
                    self.fc_layers.append(C4LinearRegularToRegular(self._full_fc_mults[i], self._full_fc_mults[i+1], bias=True))
                    if i < len(self._full_fc_mults) - 2:
                        self.fc_layers.append(act)

            def forward(self, x):
                for layer in self.conv_layers:
                    x = layer(x)

                x = self.enc_global_act(self.enc_global(x))
                x = x.view(x.shape[0], -1)                  

                for layer in self.fc_layers:
                    x = layer(x)

                return x

        # ===================== DECODER ===================== #
        class Decoder(nn.Module):
            def __init__(self, outer):
                super().__init__()
                self.conv_layers = nn.ModuleList()
                self.fc_layers = nn.ModuleList()

                act = outer.activation_function
                C = outer.C

                C_enc_mult, H_enc, W_enc = outer._enc_feat_shape
                assert outer.decoder_channels[0] == C_enc_mult, \
                    f"Expected decoder_channels[0]==encoder_channels[-1]=={C_enc_mult}, got {outer.decoder_channels[0]}"

                # 0D FC back: multiplicities
                self._full_fc_mults = list(outer.decoder_fully_connected_layers_sizes) + [C_enc_mult]
                for i in range(len(self._full_fc_mults) - 1):
                    self.fc_layers.append(C4LinearRegularToRegular(self._full_fc_mults[i], self._full_fc_mults[i+1], bias=True))
                    if i < len(self._full_fc_mults) - 2:
                        self.fc_layers.append(act)

                # global deconv: 1x1 -> (H_enc, W_enc)
                self.dec_first = C4ConvTranspose2d_RegularToRegular(
                    cin_mult=C_enc_mult,
                    cout_mult=C_enc_mult,
                    kernel_size=H_enc,
                    stride=1,
                    padding=0,
                    output_padding=0,
                    bias=True
                )
                self.dec_first_act = act

                # Upsampling blocks (like UpsamplingCNNAutoencoder2D style): Upsample -> Conv
                for i in range(outer.number_of_convolutional_layers_decoder):
                    k = outer.decoder_kernel_sizes[i]
                    p = outer.decoder_paddings[i]
                    s = outer.decoder_strides[i]

                    self.conv_layers.append(C4Upsample(scale_factor=s, mode="nearest"))

                    in_mult = outer.decoder_channels[i]
                    if i == outer.number_of_convolutional_layers_decoder - 1:
                        # last: regular -> trivial (C channels)
                        self.conv_layers.append(
                            C4Conv2d_RegularToTrivial(in_mult, C, kernel_size=k, stride=1, padding=p, bias=True)
                        )
                    else:
                        out_mult = outer.decoder_channels[i+1]
                        self.conv_layers.append(
                            C4Conv2d_RegularToRegular(in_mult, out_mult, kernel_size=k, stride=1, padding=p, bias=True)
                        )
                        self.conv_layers.append(act)

            def forward(self, x):
                # FC
                for layer in self.fc_layers:
                    x = layer(x)

                # reshape to (B, mult*4, 1, 1)
                B, Ch = x.shape
                x = x.view(B, Ch, 1, 1)

                # global deconv to (H_enc, W_enc)
                x = self.dec_first_act(self.dec_first(x))

                # upsample+conv blocks
                for layer in self.conv_layers:
                    x = layer(x)

                return x

        self.encoder = Encoder(self)
        self.decoder = Decoder(self)

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def print_parameters(self):
        print("=> Parameters of neural network:")
        print("Encoder:")
        print(f"  Convolutional layers: {int(self.number_of_convolutional_layers_encoder)}")
        print("Decoder:")
        print(f"  Convolutional layers: {int(self.number_of_convolutional_layers_decoder)}")
        print("Architecture:")
        print(self)



#############################################################################
 
 
def _bilinear_rotation_stencil(H, W, angle):
    """Indices and weights of bilinear interpolation for a rotation by `angle` about the array centre.
 
    Output pixel (i, j) samples the periodic extension of the input at
        row = c_r + (i - c_r) cos(angle) + (j - c_c) sin(angle),
        col = c_c - (i - c_r) sin(angle) + (j - c_c) cos(angle),
    so that angle = pi/2 reproduces torch.rot90(x, 1, dims=(-2, -1)) (escnn's generator of C4).
    Returns flat input indices (4, H*W) and weights (4, H*W).
    """
    c_r, c_c = 0.5 * (H - 1), 0.5 * (W - 1)
    i = torch.arange(H, dtype=torch.float64).view(H, 1) - c_r
    j = torch.arange(W, dtype=torch.float64).view(1, W) - c_c
    ca, sa = math.cos(angle), math.sin(angle)
    rows = c_r + i * ca + j * sa
    cols = c_c - i * sa + j * ca
    r0, c0 = torch.floor(rows), torch.floor(cols)
    fr, fc = rows - r0, cols - c0
    r0, c0 = r0.long(), c0.long()
 
    indices, weights = [], []
    for dr, wr in ((0, 1.0 - fr), (1, fr)):
        for dc, wc in ((0, 1.0 - fc), (1, fc)):
            indices.append((((r0 + dr) % H) * W + (c0 + dc) % W).reshape(-1))
            weights.append((wr * wc).reshape(-1))
    return torch.stack(indices), torch.stack(weights)
 
 
def _rotate_periodic_interpolated(x, angle):
    """Rotate scalar fields x of shape (..., H, W) by `angle` (radians) about the array centre.
 
    Bilinear interpolation of the periodic extension of x, written as a gather so that forward-mode AD
    (torch.func.jacfwd) and vmap work. Only needed for rotations that are not multiples of 90 degrees
    (odd elements of C8). These are NOT symmetries of the periodic pixel grid: the map is neither a
    permutation nor symplectic, and a rotated periodic field is in general no longer periodic.
    """
    H, W = x.shape[-2], x.shape[-1]
    indices, weights = _bilinear_rotation_stencil(H, W, angle)
    indices = indices.to(x.device)
    weights = weights.to(device=x.device, dtype=x.dtype)
    x_flat = x.reshape(*x.shape[:-2], H * W)
    x_rot = sum(weights[m] * x_flat[..., indices[m]] for m in range(4))
    return x_rot.reshape(x.shape)
 
 
def _rotate_cn(x, k, N):
    """Group action rho_X(g_k) of the rotation g_k by 2*pi*k/N on scalar fields x of shape (B, C, H, W).
 
    Matches the convention of escnn's gspaces.rot2dOnR2(N): the generator of C4 acts as
    torch.rot90(x, 1, dims=(-2, -1)). Multiples of 90 degrees are exact pixel permutations
    (rotation about the array centre, as np.rot90 in the data pipeline); any remaining angle
    (45 degrees for odd elements of C8) is applied by interpolation.
    """
    k = int(k) % N
    quarter_turns, remainder = divmod(4 * k, N)
    if remainder:
        x = _rotate_periodic_interpolated(x, 0.5 * math.pi * remainder / N)
    if quarter_turns:
        x = torch.rot90(x, quarter_turns, dims=(-2, -1))
    return x
 
 
class InvariantPoseGCNNAutoencoder2D(nn.Module):
    """Invariant-code / pose autoencoder following Winter et al. (NeurIPS 2022), for H = C_N.
 
    The reconstruction is x_hat = rho_X(psi(x)) delta(eta(x)) with
        eta   : G-invariant encoder (escnn group convolutions, group pooling at the 1x1 bottleneck,
                then plain linear layers) -> latent code z of size encoder_fully_connected_layers_sizes[-1],
        psi   : pose, psi = chi o mu with mu G-equivariant (escnn) and chi deterministic,
        delta : plain PyTorch decoder (same structure as UpsamplingCNNAutoencoder2D) producing the
                state in canonical orientation.
 
    mu outputs pose scores s(x) in R^N with s(g_k x) = P^k s(x) (cyclic shift), and chi(s) = g_argmax(s).
      - pose_representation='regular': s is one regular field of C_N.
      - pose_representation='irrep'  : mu outputs one frequency-1 field v in R^2 (rotates as rho_1), and
        s_k = <v, rho_1(g_k) e_1>, so chi(v) is the angle of v rounded to the nearest multiple of 2*pi/N.
        This is Winter et al.'s SO(2) construction (Eq. 4) restricted to C_N.
    Unlike the regular-representation GCNN-AE, the latent code has exactly the requested size; the
    orientation is stored once, as the discrete pose, and does not enter the reduced coordinates.
 
    Training (forward in train mode) combines the N rotated reconstructions with weights w(x):
      - pose_relaxation='hard'            : w = one_hot(argmax s). mu receives no gradient from the
                                            reconstruction and is trained by the pose loss only.
      - pose_relaxation='straight_through': forward as 'hard', gradient of softmax(s / pose_temperature).
      - pose_relaxation='soft'            : w = softmax(s / pose_temperature) (Winter-style, differentiable).
    With pose_loss_weight > 0, forward additionally stores the cross-entropy between softmax(s / T) and the
    identity pose, retrievable via auxiliary_loss() (the Trainer adds it to the loss). This fixes the
    canonical orientation to the orientation of the training data and is only meaningful if all training
    snapshots share that orientation.
 
    Interface for the ROM utilities: encode(x) returns only the invariant code z, and decode(z) /
    self.decoder(z) apply the stored pose (set_pose, default identity). The ROM of a rotated problem is
    obtained by setting its pose once before time stepping; the reduced dynamics are unchanged.
    """
    def __init__(self,
                 dims=None,
                 encoder_channels=None,
                 decoder_channels=None,
                 encoder_fully_connected_layers_sizes=None,
                 decoder_fully_connected_layers_sizes=None,
                 gspace=gspaces.rot2dOnR2(N=4),
                 activation_function=ELU,
                 torch_activation_function=nn.ELU(),
                 encoder_kernel_sizes=5,
                 encoder_paddings=2,
                 encoder_strides=2,
                 decoder_kernel_sizes=5,
                 decoder_paddings=2,
                 decoder_strides=2,
                 pose_representation='regular',
                 pose_encoder_channels=None,
                 pose_kernel_sizes=3,
                 pose_paddings=1,
                 pose_strides=2,
                 pose_relaxation='soft',
                 pose_temperature=0.05,
                 pose_loss_weight=1e-3):
 
        super(InvariantPoseGCNNAutoencoder2D, self).__init__()
 
        def _as_list(v, n):
            if isinstance(v, (list, tuple)):
                assert len(v) == n, f"Expected {n} values, got {len(v)}"
                return list(v)
            return [int(v)] * n
 
        assert pose_representation in ('regular', 'irrep'), f"Unknown pose_representation '{pose_representation}'"
        assert pose_relaxation in ('hard', 'straight_through', 'soft'), f"Unknown pose_relaxation '{pose_relaxation}'"
        assert int(dims[1]) == int(dims[2]), "Rotations by 90 degrees require a square grid (Nx == Ny)"
 
        self.number_of_convolutional_layers_encoder = len(encoder_channels)
        self.encoder_channels = encoder_channels
        self.number_of_fully_connected_layers_encoder = len(encoder_fully_connected_layers_sizes)
        self.encoder_fully_connected_layers_sizes = encoder_fully_connected_layers_sizes
        self.number_of_convolutional_layers_decoder = len(decoder_channels)
        self.decoder_channels = decoder_channels
        self.number_of_fully_connected_layers_decoder = len(decoder_fully_connected_layers_sizes)
        self.decoder_fully_connected_layers_sizes = decoder_fully_connected_layers_sizes
 
        self.gspace = gspace
        self.group_order = int(gspace.fibergroup.order())
        self.activation_function = activation_function
        self.torch_activation_function = torch_activation_function
 
        self.C = int(dims[0])
        self.Nx = int(dims[1])
        self.Ny = int(dims[2])
 
        self.encoder_kernel_sizes = _as_list(encoder_kernel_sizes, self.number_of_convolutional_layers_encoder)
        self.encoder_paddings = _as_list(encoder_paddings, self.number_of_convolutional_layers_encoder)
        self.encoder_strides = _as_list(encoder_strides, self.number_of_convolutional_layers_encoder)
 
        self.decoder_kernel_sizes = _as_list(decoder_kernel_sizes, self.number_of_convolutional_layers_decoder)
        self.decoder_paddings = _as_list(decoder_paddings, self.number_of_convolutional_layers_decoder)
        self.decoder_strides = _as_list(decoder_strides, self.number_of_convolutional_layers_decoder)
 
        # pose network mu: shares the escnn trunk of eta unless its own channels are given
        self.share_pose_backbone = pose_encoder_channels is None
        self.pose_encoder_channels = pose_encoder_channels
        if not self.share_pose_backbone:
            self.number_of_convolutional_layers_pose = len(pose_encoder_channels)
            self.pose_kernel_sizes = _as_list(pose_kernel_sizes, self.number_of_convolutional_layers_pose)
            self.pose_paddings = _as_list(pose_paddings, self.number_of_convolutional_layers_pose)
            self.pose_strides = _as_list(pose_strides, self.number_of_convolutional_layers_pose)
 
        self.pose_representation = pose_representation
        self.pose_relaxation = pose_relaxation
        self.pose_temperature = float(pose_temperature)
        self.pose_loss_weight = float(pose_loss_weight)
        self._auxiliary_loss = None

        self.eps = 1e-6
 
        def conv_out(n, kernel_size=5, padding=2, stride=2, dilation=1):
            return ((n + 2*padding - dilation*(kernel_size-1) - 1)//stride) + 1
 
        def build_equivariant_trunk(module, channels, kernel_sizes, paddings, strides):
            """escnn conv stack trivial -> regular -> ... followed by a global conv to a 1x1 map of regular fields."""
            module.conv_layers = nn.ModuleList()
            H, W = self.Ny, self.Nx
            module.spatial_sizes = []   # spatial size at the input of every conv layer
            in_type = FieldType(self.gspace, self.C * [self.gspace.trivial_repr])
            for i in range(len(channels)):
                k, p, s = kernel_sizes[i], paddings[i], strides[i]
                module.spatial_sizes.append((H, W))
                out_type = FieldType(self.gspace, channels[i] * [self.gspace.regular_repr])
                module.conv_layers.extend([R2Conv(in_type, out_type, kernel_size=k, padding=p, stride=s, padding_mode='circular', bias=True)])
                module.conv_layers.extend([self.activation_function(out_type, inplace=False)])
                in_type = out_type
                H = conv_out(H, kernel_size=k, padding=p, stride=s)
                W = conv_out(W, kernel_size=k, padding=p, stride=s)
 
            # Global equivariant conv to 1x1
            assert H == W, "Global convolution to 1x1 requires a square feature map"
            module.global_conv = R2Conv(in_type, FieldType(self.gspace, channels[-1] * [self.gspace.regular_repr]), kernel_size=H, stride=1, padding=0, bias=True)
            module.global_act = self.activation_function(module.global_conv.out_type, inplace=False)
            return H, W
 
        def trunk_forward(module, x):
            x = GeometricTensor(x, FieldType(self.gspace, self.C * [self.gspace.trivial_repr]))
            for layer in module.conv_layers:
                x = layer(x)
            return module.global_act(module.global_conv(x))
 
        # ===================== INVARIANT ENCODER (eta) ===================== #
        class InvariantEncoder(nn.Module):
            def __init__(self, outer):
                super(InvariantEncoder, self).__init__()
                self.fc_layers = nn.ModuleList()
 
                self.H, self.W = build_equivariant_trunk(self, outer.encoder_channels, outer.encoder_kernel_sizes, outer.encoder_paddings, outer.encoder_strides)
                outer._enc_feat_shape = (outer.encoder_channels[-1], self.H, self.W)
                outer._enc_spatial_sizes = list(self.spatial_sizes)
 
                # Max over the group channels of every regular field -> invariant scalars
                # (done by hand in head(): escnn's GroupPooling returns float32 even for double inputs)
                self.number_of_fields = outer.encoder_channels[-1]
                self.group_order = outer.group_order
 
                # Invariant features: plain linear layers keep invariance
                self._full_fc_sizes = [outer.encoder_channels[-1]] + list(outer.encoder_fully_connected_layers_sizes)
                for i in range(len(self._full_fc_sizes) - 1):
                    self.fc_layers.extend([nn.Linear(self._full_fc_sizes[i], self._full_fc_sizes[i+1])])
                    if i < len(self._full_fc_sizes) - 2:
                        self.fc_layers.extend([outer.torch_activation_function])
 
            def trunk(self, x):
                return trunk_forward(self, x)
 
            def head(self, h):
                # regular fields are stored field-major: (B, fields * N, 1, 1) -> (B, fields, N) -> max over N
                x = h.tensor.view(h.tensor.shape[0], self.number_of_fields, self.group_order).amax(dim=2)
                for layer in self.fc_layers:
                    x = layer(x)
                return x
 
            def forward(self, x):
                return self.head(self.trunk(x))
 
        # ===================== POSE ENCODER (mu) ===================== #
        class PoseEncoder(nn.Module):
            def __init__(self, outer):
                super(PoseEncoder, self).__init__()
                N = outer.group_order
 
                if outer.share_pose_backbone:
                    in_mult = outer.encoder_channels[-1]
                else:
                    build_equivariant_trunk(self, outer.pose_encoder_channels, outer.pose_kernel_sizes,
                                            outer.pose_paddings, outer.pose_strides)
                    in_mult = outer.pose_encoder_channels[-1]
 
                # equivariant linear map on the 0D gspace: regular fields -> pose field
                self.gspace0d = gspaces.no_base_space(outer.gspace.fibergroup)
                self.in_type0 = FieldType(self.gspace0d, in_mult * [self.gspace0d.regular_repr])
                if outer.pose_representation == 'regular':
                    out_type0 = FieldType(self.gspace0d, [self.gspace0d.regular_repr])
                else:
                    out_type0 = FieldType(self.gspace0d, [self.gspace0d.irrep(1)])
                    # directions rho_1(g_k) e_1, k = 0, ..., N-1
                    rho_1 = self.gspace0d.irrep(1)
                    directions = [[float(v) for v in rho_1(outer.gspace.fibergroup.element(k))[:, 0]] for k in range(N)]
                    self.register_buffer("directions", torch.tensor(directions))
                self.pose_layer = Linear(self.in_type0, out_type0, bias=True)
                self.pose_representation = outer.pose_representation
 
            def trunk(self, x):
                return trunk_forward(self, x)
 
            def head(self, h):
                B = h.tensor.shape[0]
                dtype = next(self.pose_layer.parameters()).dtype
                with torch.autocast(device_type=h.tensor.device.type, enabled=False):
                    x = GeometricTensor(h.tensor.view(B, -1).to(dtype), self.in_type0)
                    x = self.pose_layer(x).tensor
                    # Normalize the scores with invariant statistics, which keeps chi = argmax equivariant and bounds
                    # the logits: otherwise the pose loss and the soft relaxation push them apart without limit (Adam
                    # keeps making steps of size ~lr even when the cross-entropy is already ~0).
                    # rsqrt(. + eps) keeps the gradient finite for inputs with tied scores (e.g. constant fields).
                    if self.pose_representation == 'irrep':
                        # unit vector on S^1, the homogeneous space of Winter et al.'s SO(2) construction
                        x = x * torch.rsqrt(x.pow(2).sum(dim=1, keepdim=True) + self.eps)
                        x = x @ self.directions.t()
                    else:
                        # zero mean and unit standard deviation over the N group channels (both invariant under C_N)
                        x = x - x.mean(dim=1, keepdim=True)
                        x = x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)
                return x

 
        # ===================== DECODER (delta) ===================== #
        class CanonicalDecoder(nn.Module):
 
            def __init__(self, outer):
                super(CanonicalDecoder, self).__init__()
                self.conv_layers = nn.ModuleList()
                self.fc_layers = nn.ModuleList()
 
                activation_function = outer.torch_activation_function
                C = outer.C
                self.decoder_channels = outer.decoder_channels
 
                C_enc, H_enc, W_enc = outer._enc_feat_shape
                self.start_cl = (H_enc, W_enc)
                assert outer.decoder_channels[0] == C_enc
 
                # compute size of last fc layer automatically
                self._full_fc_sizes = list(outer.decoder_fully_connected_layers_sizes) + [outer.decoder_channels[0]]
                assert self._full_fc_sizes[0] == outer.encoder_fully_connected_layers_sizes[-1], \
                    "decoder_fully_connected_layers_sizes[0] must equal the latent dimension"
 
                # Fully connected layers
                for i in range(len(self._full_fc_sizes) - 1):
                    self.fc_layers.extend([nn.Linear(self._full_fc_sizes[i], self._full_fc_sizes[i+1])])
                    if i < len(self._full_fc_sizes) - 2:
                        self.fc_layers.extend([activation_function])
 
                self.dec_first = nn.ConvTranspose2d(outer.encoder_channels[-1], outer.encoder_channels[-1], kernel_size=H_enc, stride=1, padding=0, output_padding=0, bias=True)
                self.dec_first_act = activation_function
 
                # If the decoder mirrors the encoder, upsample to the mirrored encoder sizes (identical to
                # scale_factor=2 for 256 -> ... -> 8, but also works for odd grids such as 257 -> ... -> 9)
                mirrored_sizes = list(reversed(outer._enc_spatial_sizes))
                mirror_encoder = len(mirrored_sizes) == outer.number_of_convolutional_layers_decoder
 
                # Upsampling + convolutional layers
                for i in range(outer.number_of_convolutional_layers_decoder):
                    k = outer.decoder_kernel_sizes[i]
                    p = outer.decoder_paddings[i]
                    s = outer.decoder_strides[i]
                    upsampling = nn.Upsample(size=mirrored_sizes[i]) if mirror_encoder else nn.Upsample(scale_factor=s)
 
                    if i == outer.number_of_convolutional_layers_decoder - 1:
                        self.conv_layers.extend([upsampling])
                        self.conv_layers.extend([nn.Conv2d(decoder_channels[i], C, kernel_size=k, padding=p, stride=1, padding_mode='circular', bias=True)])
                    else:
                        self.conv_layers.extend([upsampling])
                        self.conv_layers.extend([nn.Conv2d(decoder_channels[i], decoder_channels[i+1], kernel_size=k, padding=p, stride=1, padding_mode='circular', bias=True)])
 
                    if i != outer.number_of_convolutional_layers_decoder - 1:
                        self.conv_layers.extend([activation_function])
 
            def forward(self, x):
                for layer in self.fc_layers:
                    x = layer(x)
 
                B, _ = x.shape
                x = x.view(B, self.decoder_channels[0], 1, 1)
                x = self.dec_first_act(self.dec_first(x))
 
                for layer in self.conv_layers:
                    x = layer(x)
 
                return x
 
        class PosedDecoder(nn.Module):
            """z -> rho_X(g_pose) delta(z) with a fixed pose; this is the map used in the ROM."""
            def __init__(self, outer):
                super(PosedDecoder, self).__init__()
                self.canonical = CanonicalDecoder(outer)
                self.group_order = outer.group_order
                self.register_buffer("pose", torch.zeros((), dtype=torch.long), persistent=False)
 
            def forward(self, x):
                return _rotate_cn(self.canonical(x), int(self.pose), self.group_order)
 
        self.encoder = InvariantEncoder(self)
        self.pose_encoder = PoseEncoder(self)
        self.decoder = PosedDecoder(self)
 
    def _code_and_pose_scores(self, x):
        h = self.encoder.trunk(x)
        z = self.encoder.head(h)
        h_pose = h if self.share_pose_backbone else self.pose_encoder.trunk(x)
        return z, self.pose_encoder.head(h_pose)
 
    def _all_rotations(self, x):
        """Stack rho_X(g_k) x for k = 0, ..., N-1 along dim 1 -> (B, N, C, H, W)."""
        return torch.stack([_rotate_cn(x, k, self.group_order) for k in range(self.group_order)], dim=1)
 
    def _rotate_per_sample(self, x, poses):
        return torch.cat([_rotate_cn(x[b:b+1], int(poses[b]), self.group_order) for b in range(x.shape[0])], dim=0)
 
    def forward(self, x):
        z, scores = self._code_and_pose_scores(x)
        canonical = self.decoder.canonical(z)
        poses = scores.argmax(dim=1)
 
        if self.training and self.pose_relaxation != 'hard':
            probabilities = F.softmax(scores / self.pose_temperature, dim=1)
            if self.pose_relaxation == 'straight_through':
                one_hot = F.one_hot(poses, self.group_order).to(probabilities.dtype)
                weights = one_hot + probabilities - probabilities.detach()
            else:
                weights = probabilities
            decoded = torch.einsum('bk,bkchw->bchw', weights, self._all_rotations(canonical))
        else:
            decoded = self._rotate_per_sample(canonical, poses)
 
        if self.pose_loss_weight > 0:
            identity = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            self._auxiliary_loss = self.pose_loss_weight * F.cross_entropy(scores / self.pose_temperature, identity)
        else:
            self._auxiliary_loss = None
 
        return decoded
 
    def auxiliary_loss(self):
        """Weighted pose loss of the last forward pass (zero if disabled)."""
        if self._auxiliary_loss is None:
            return torch.zeros((), dtype=next(self.parameters()).dtype, device=next(self.parameters()).device)
        return self._auxiliary_loss
 
    def print_parameters(self):
        print("=> Parameters of neural network:")
        print(f"Group: C{self.group_order}, pose representation: {self.pose_representation}, pose relaxation: {self.pose_relaxation}")
        print("Encoder:")
        print(f'Convolutional layers: {int(self.number_of_convolutional_layers_encoder)}')
        print(f'Pose network shares encoder backbone: {self.share_pose_backbone}')
        print("Decoder:")
        print(f'Convolutional layers: {int(self.number_of_convolutional_layers_decoder)}')
        print("Architecture:")
        print(self)
 
    def encode(self, x):
        """Invariant code z = eta(x)."""
        return self.encoder(x)
 
    def pose_scores(self, x):
        """Equivariant pose scores s(x) in R^N (mu composed with the frequency projection for 'irrep')."""
        return self._code_and_pose_scores(x)[1]
 
    def estimate_pose(self, x):
        """Pose index k with psi(x) = g_k, i.e. chi(mu(x)) = argmax of the pose scores."""
        return self.pose_scores(x).argmax(dim=1)
 
    def encode_with_pose(self, x):
        z, scores = self._code_and_pose_scores(x)
        return z, scores.argmax(dim=1)
 
    def set_pose(self, pose):
        """Fix the pose used by decode(z) / self.decoder(z), e.g. once per ROM trajectory."""
        pose = int(pose) % self.group_order
        if 4 * pose % self.group_order != 0:
            print(f"Warning: pose {pose} of C{self.group_order} is not a multiple of 90 degrees; "
                  "the decoder rotation is interpolated and not an exact symmetry of the grid.")
        self.decoder.pose.fill_(pose)
 
    def decode_canonical(self, z):
        """delta(z): reconstruction in canonical orientation."""
        return self.decoder.canonical(z)
 
    def decode(self, z, pose=None):
        """rho_X(g_pose) delta(z). Uses the stored pose (set_pose) if pose is None; pose may be an int or a (B,) tensor."""
        if pose is None:
            return self.decoder(z)
        if isinstance(pose, int):
            return _rotate_cn(self.decoder.canonical(z), pose, self.group_order)
        return self._rotate_per_sample(self.decoder.canonical(z), pose)

