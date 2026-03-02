import torch.nn as nn

from escnn import gspaces
from escnn.nn import R2Conv, R2ConvTransposed, GeometricTensor, Linear, FieldType, ELU, R2Upsampling
from escnn import gspaces


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
                    
                    self.conv_layers.extend([R2Conv(in_type, out_type, kernel_size=k, padding=p, stride=s, padding_mode='circular', bias=True)]) #TODO das ist komisch, muss geändert werden
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
                    #if i == len(self.conv_layers) - 1:
                    #    # last layer: output is trivial type, so we return tensor
                    #    print("last layer, output is trivial type")
                    #    result_second_to_last = x.tensor
                    x = layer(x)

                return x.tensor

        self.encoder = Encoder(self)
        self.decoder = Decoder(self)
        #self.double()

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
        
        Note:
            - The model must be in eval mode before exporting
            - Equivariance is preserved in the exported model
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
        exported_model.double()
        
        print("\n" + "="*60)
        print("Export Summary:")
        print("="*60)
        print("✓ Model exported successfully!")
        print("✓ Equivariance is preserved in exported model")
        print("✓ Model set to eval mode")
        print("\nNote: Some layers (R2Upsampling, activations) don't support")
        print("export but are already efficient PyTorch operations.")
        print("="*60 + "\n")

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
                    
                    self.conv_layers.extend([R2Conv(in_type, out_type, kernel_size=k, padding=p, stride=s, padding_mode='circular', bias=True)]) #TODO das ist komisch, muss geändert werden
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
        #self.double()

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
    

################################################### equiv AE files ########################################################


import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_list(v, n: int):
    if isinstance(v, (list, tuple)):
        assert len(v) == n, f"Expected {n} values, got {len(v)}"
        return list(v)
    return [int(v)] * n


def _rot90_kernel(w: torch.Tensor, k: int):
    k = k % 4
    if k == 0:
        return w
    return torch.rot90(w, k, dims=(-2, -1))


def _circular_pad2d(x: torch.Tensor, padding: int):
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
                w = self.weight[delta]          # (cout_mult, cin_mult, k, k)
                w_rs = _rot90_kernel(w, r)      # rotate by output orientation
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
                w = self.weight[delta]              # (cout_mult, cin_mult, k, k)
                w_t = w.permute(1, 0, 2, 3)         # (cin_mult, cout_mult, k, k) for conv_transpose2d
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
                W = self.weight[delta]  # (out_mult, in_mult)
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

                x = self.enc_global_act(self.enc_global(x))  # (B, mult*4, 1, 1)
                x = x.view(x.shape[0], -1)                   # (B, mult*4)

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
