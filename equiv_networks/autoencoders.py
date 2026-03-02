import torch.nn as nn

from escnn import gspaces
from escnn.nn import R2Conv, R2ConvTransposed, GeometricTensor, ReLU, Linear, FieldType, PointwiseAvgPool2D, ELU, R2Upsampling
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

                # compute the values for output padding such that the output size is again Nx x Ny
                #ops = self.plan_output_padding(
                #    target=outer.Ny,
                #    layers=outer.number_of_convolutional_layers_decoder,
                #    start=H_enc,
                #    ks=outer.decoder_kernel_sizes,
                #    ps=outer.decoder_paddings,
                #    ss=outer.decoder_strides,
                #    ds=[1]*outer.number_of_convolutional_layers_decoder
                #)

                #self._output_padding_list = ops
                self.dec_first = nn.ConvTranspose2d(outer.encoder_channels[-1], outer.encoder_channels[-1], kernel_size=H_enc, stride=1, padding=0, output_padding=0, bias=True)
                self.dec_first_act = activation_function
                
                # Transposed convolutional layers
                for i in range(outer.number_of_convolutional_layers_decoder):
                    k = outer.decoder_kernel_sizes[i]
                    p = outer.decoder_paddings[i]
                    s = outer.decoder_strides[i]
                    #op = self._output_padding_list[i]

                    if i == outer.number_of_convolutional_layers_decoder - 1:
                        #self.conv_layers.extend([nn.ConvTranspose2d(decoder_channels[i], C, kernel_size=k, stride=s, padding=p, output_padding=op)])
                        self.conv_layers.extend([nn.Upsample(scale_factor=s)])
                        self.conv_layers.extend([nn.Conv2d(decoder_channels[i], C, kernel_size=k, padding=p, stride=1, padding_mode='circular', bias=True)])
                    else:
                        #self.conv_layers.extend([nn.ConvTranspose2d(decoder_channels[i], decoder_channels[i+1], kernel_size=k, stride=s, padding=p, output_padding=op)])
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


######################################################################################################


class RotationGCNNAutoencoder2D(nn.Module):
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

        super(RotationGCNNAutoencoder2D, self).__init__()

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

        # store pose from last encode() for decode()
        # NEW
        self._last_tau = None
        self._last_k = None

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
                    
                    self.conv_layers.extend([R2Conv(in_type, out_type, kernel_size=k, padding=p, stride=s, padding_mode='circular', bias=False)])
                    self.conv_layers.extend([self.activation_function(out_type, inplace=False)])

                    # track spatial size
                    self.H = self.conv_out(self.H, kernel_size=k, padding=p, stride=s)
                    self.W = self.conv_out(self.W, kernel_size=k, padding=p, stride=s)

                # Pose heads on small map
                #TODO das muss auskommentiert werden
                #rot_type = FieldType(self.gspace, 4 * [self.gspace.trivial_repr]) #TODO die 4 hier kommt von den channels
                #attn_type = FieldType(self.gspace, 1 * [self.gspace.trivial_repr])
                #self.rot_head = R2Conv(out_type, rot_type, kernel_size=1, padding=0, stride=1, bias=True)
                #self.attn_head = R2Conv(out_type, attn_type, kernel_size=1, padding=0, stride=1, bias=True)

                # Global equivariant conv to 1x1 (canonical bottleneck)
                self.global_in_type = out_type
                self.global_out_type = FieldType(self.gspace, outer.encoder_channels[-1] * [self.gspace.regular_repr])

                self.dec_first = R2Conv(self.global_in_type, self.global_out_type, kernel_size=self.H, stride=1, padding=0, bias=False)
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
                    #TODO hier veränderung
                    #if i < len(self._full_fc_sizes) - 2:
                    self.fc_layers.extend([self.activation_function(out_type0, inplace=False)])

            def forward(self, x):
                x = GeometricTensor(x, FieldType(self.gspace, self.C * [self.gspace.trivial_repr]))
                for layer in self.conv_layers:
                    x = layer(x)

                # pose heads
                # NEW
                #rot_logits = self.rot_head(x).tensor.mean(dim=(2,3))
                #k = rot_logits.argmax(dim=1)
                #A = self.attn_head(x).tensor
                #tau = self._soft_argmax_tau(A)

                # canonicalize
                # NEW
                #x = _rotate_C4_batch(x, -k)
                #x = _translate(x, -tau)

                # global equivariant conv
                # NEW
                x = self.dec_first_act(self.dec_first(x))
                #global_pool = PointwiseAvgPool2D(self.global_in_type, kernel_size=self.H, stride=1, padding=0).to(dtype=x.tensor.dtype, device=x.tensor.device)
                #x = global_pool(x)

                # 0D linear stack -> vector latent
                B, Ch, _, _ = x.tensor.shape
                x= GeometricTensor(x.tensor.view(B, Ch), FieldType(self.gspace0d, self._full_fc_sizes[0] * [self.gspace0d.regular_repr]))
                for layer in self.fc_layers:
                    x = layer(x)

                return x.tensor
            
            def conv_out(self, n, kernel_size=5, padding=2, stride=2, dilation=1):
                return ((n + 2*padding - dilation*(kernel_size-1) - 1)//stride) + 1

            def _soft_argmax_tau(self, A):
                B, _, H, W = A.shape
                A = A.flatten(2).softmax(dim=-1).view(B, 1, H, W)
                ys = torch.linspace(-1, 1, H, device=A.device, dtype=A.dtype)
                xs = torch.linspace(-1, 1, W, device=A.device, dtype=A.dtype)
                gy, gx = torch.meshgrid(ys, xs, indexing='ij')
                dy = (A[:, 0] * gy).sum(dim=(1, 2))
                dx = (A[:, 0] * gx).sum(dim=(1, 2))
                return torch.stack([dy, dx], dim=1)

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

                    #TODO hier veränderung
                    #if i < len(self._full_fc_sizes) - 2:
                    self.fc_layers.extend([self.activation_function(out_type0, inplace=False)])
                
                self.enc_out_type = FieldType(self.gspace, outer.encoder_channels[-1] * [self.gspace.regular_repr])
                self.dec_first = R2ConvTransposed(self.enc_out_type,
                                            FieldType(self.gspace, self.encoder_channels[-1] * [self.gspace.regular_repr]), #TODO hier zwei und eins
                                            kernel_size=H_enc,
                                            stride=1,
                                            padding=0, 
                                            output_padding=0, 
                                            bias=False)
                
                self.dec_first_act = self.activation_function(self.dec_first.out_type, inplace=False)

                ops = self.plan_output_padding(
                    target=outer.Nx,
                    layers=outer.number_of_convolutional_layers_decoder,
                    start=H_enc,
                    ks=outer.decoder_kernel_sizes,
                    ps=outer.decoder_paddings,
                    ss=outer.decoder_strides,
                    ds=[1]*outer.number_of_convolutional_layers_decoder
                )

                self._output_padding_list = ops
                
                # Mirror upsampling: use user-provided decoder lists
                # We start from the same type as encoder's last feature BEFORE global conv
                in_t = FieldType(self.gspace, outer.encoder_channels[-1] * [self.gspace.regular_repr]) #TODO hier zwei und eins
                for i in range(outer.number_of_convolutional_layers_decoder):
                    k = outer.decoder_kernel_sizes[i]
                    p = outer.decoder_paddings[i]
                    s = outer.decoder_strides[i]
                    op = self._output_padding_list[i]

                    if i == outer.number_of_convolutional_layers_decoder - 1:
                        out_t = FieldType(self.gspace,self.C * [self.gspace.trivial_repr])
                    else:
                        out_t = FieldType(self.gspace, outer.decoder_channels[i+1] * [self.gspace.regular_repr])

                    self.conv_layers.extend([R2ConvTransposed(in_t, out_t, kernel_size=k, padding=p, stride=s, output_padding=op, bias=False)])

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
                for layer in self.conv_layers:
                    x = layer(x)

                # inverse pose: rotate +k, translate +tau
                #x = _rotate_C4_batch(x, k)
                #x = _translate(x, tau)

                return x.tensor
            
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
                    for op in (0, 1 if s > 1 else 0,):
                        n2 = self.deconv_out(n, k, p, s, d, op)
                        dq.append((n2, ops + [op]))
                raise ValueError(f"No output_padding sequence found for target={target} with layers={layers}.")
  
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
        z = self.encoder(x)
        #self._last_tau = tau
        #self._last_k = k
        return z
  
    def decode(self, x):
        #assert self._last_tau is not None and self._last_k is not None, "encode(...) must be called before decode(...)."
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

    
# escnn based autoencoder with trivial representations only


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
    

    

