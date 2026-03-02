import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Helpers (same as before)
# ============================================================

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


# ============================================================
# C4 blocks (same as before)
# Channel convention: (B, mult*4, H, W)
# ============================================================

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


# ============================================================
# Torch-only AE with Encoder/Decoder inner classes (your style)
# ============================================================

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
                 gspace=None,  # kept for drop-in signature compatibility
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
