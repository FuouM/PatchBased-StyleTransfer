from typing import OrderedDict
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
from torch.nn.utils import spectral_norm


class UpsamplingLayer(nn.Module):
    def __init__(self, channels):
        super(UpsamplingLayer, self).__init__()
        self.layer = nn.Upsample(scale_factor=2)

    def forward(self, x):
        return self.layer(x)


#####
# Currently default generator we use
# conv0 -> conv1 -> conv2 -> resnet_blocks -> upconv2 -> upconv1 ->  conv_11 -> (conv_11_a)* -> conv_12 -> (Tanh)*
# there are 2 conv layers inside conv_11_a
# * means is optional, model uses skip-connections
class GeneratorJ(nn.Module):
    def make_layers(self):
        sizes = [
            7,
            3,
            3,
        ]
        strides = [
            1,
            2,
            2,
        ]
        paddings = [
            3,
            1,
            1,
        ]
        in_filters = [
            self.input_channels,
            self.filters[0],
            self.filters[1],
        ]
        out_filters = [
            self.filters[0],
            self.filters[1],
            self.filters[2],
        ]

        for i in range(3):
            setattr(
                self,
                f"conv{i}",
                self.relu_layer(
                    in_filters=in_filters[i],
                    out_filters=out_filters[i],
                    size=sizes[i],
                    stride=strides[i],
                    padding=paddings[i],
                    bias=self.use_bias,
                    norm_layer=self.norm_layer,
                    nonlinearity=nn.LeakyReLU(0.2),
                ),
            )

        for i in range(self.resnet_blocks):
            self.resnets.append(
                self.resnet_block(
                    in_filters=self.filters[2],
                    out_filters=self.filters[2],
                    size=3,
                    stride=1,
                    padding=1,
                    bias=self.use_bias,
                    norm_layer=self.norm_layer,
                    nonlinearity=nn.ReLU(),
                )
            )

    def __init__(
        self,
        input_size=256,
        norm_layer="batch_norm",
        gpu_ids=None,
        use_bias=False,
        resnet_blocks=9,
        tanh=False,
        filters=(64, 128, 128, 128, 128, 64),
        input_channels=3,
        append_smoothers=False,
        use_spectral_norm=False,
        skip=True
    ):
        super(GeneratorJ, self).__init__()
        self.input_size = input_size
        assert norm_layer in [
            None,
            "batch_norm",
            "instance_norm",
        ], "norm_layer should be None, 'batch_norm' or 'instance_norm', not {}".format(
            norm_layer
        )
        self.norm_layer = None
        if norm_layer == "batch_norm":
            self.norm_layer = nn.BatchNorm2d
        elif norm_layer == "instance_norm":
            self.norm_layer = nn.InstanceNorm2d
        self.input_channels = input_channels
        self.filters = filters
        self.gpu_ids = gpu_ids
        self.use_bias = use_bias
        self.resnet_blocks = resnet_blocks
        self.append_smoothers = append_smoothers
        self.use_spectral_norm = use_spectral_norm
        self.skip = skip

        self.resnets = nn.ModuleList()
        self.make_layers()

        self.upconv2 = self.upconv_layer_upsample_and_conv(
            in_filters=filters[3] + filters[2] if self.skip else filters[3],
            out_filters=filters[4],
            size=4,
            stride=2,
            padding=1,
            bias=self.use_bias,
            norm_layer=self.norm_layer,
            nonlinearity=nn.ReLU(),
        )

        self.upconv1 = self.upconv_layer_upsample_and_conv(
            in_filters=filters[4] + filters[1] if self.skip else filters[4],
            out_filters=filters[4],
            size=4,
            stride=2,
            padding=1,
            bias=self.use_bias,
            norm_layer=self.norm_layer,
            nonlinearity=nn.ReLU(),
        )

        self.conv_11 = nn.Sequential(
            nn.Conv2d(
                in_channels=filters[0] + filters[4] + input_channels if self.skip else filters[4],
                out_channels=filters[5],
                kernel_size=7,
                stride=1,
                padding=3,
                bias=self.use_bias,
            ),
            nn.ReLU(),
        )

        if self.append_smoothers:
            self.conv_11_a = nn.Sequential(
                nn.Conv2d(
                    filters[5], filters[5], kernel_size=3, bias=self.use_bias, padding=1
                ),
                nn.ReLU(),
                nn.BatchNorm2d(num_features=filters[5]),  # replace with variable
                nn.Conv2d(
                    filters[5], filters[5], kernel_size=3, bias=self.use_bias, padding=1
                ),
                nn.ReLU(),
            )

        if tanh:
            self.conv_12 = nn.Sequential(
                nn.Conv2d(filters[5], 3, kernel_size=1, stride=1, padding=0, bias=True),
                # nn.Tanh(), #[-1, 1]
                nn.Sigmoid(),  # According to the Lightning implementation, [0, 1]
            )
        else:
            self.conv_12 = nn.Conv2d(
                filters[5], 3, kernel_size=1, stride=1, padding=0, bias=True
            )

    def forward(self, x):
        output_0 = self.conv0(x) # [21, 32, 128, 128]
        output_1 = self.conv1(output_0) # [21, 64, 64, 64]
        output_2 = self.conv2(output_1)  # Store the output # [21, 128, 32, 32]
        output = output_2  # comment to disable skip-connections # [21, 128, 32, 32]
        for layer in self.resnets:
            output = layer(output) + output

        if self.skip:
            output = self.upconv2(torch.cat((output, output_2), dim=1)) # [21, 128, 32, 32]
            output = self.upconv1(torch.cat((output, output_1), dim=1)) # [21, 128, 32, 32]
            output = self.conv_11(torch.cat((output, output_0, x), dim=1)) # [21, 128, 32, 32]
        else:
            output = self.upconv2(output)  # disable skip-connections
            output = self.upconv1(output)  # disable skip-connections
            output = self.conv_11(output)  # disable skip-connections

        if self.append_smoothers:
            output = self.conv_11_a(output)
            
        output = self.conv_12(output)
        return output

    def relu_layer(
        self,
        in_filters,
        out_filters,
        size,
        stride,
        padding,
        bias,
        norm_layer,
        nonlinearity,
    ):
        out = nn.Sequential()

        conv = nn.Conv2d(
            in_channels=in_filters,
            out_channels=out_filters,
            kernel_size=size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        if self.use_spectral_norm:
            conv = spectral_norm(conv)

        out.add_module("conv", conv)

        if norm_layer:
            out.add_module("normalization", norm_layer(num_features=out_filters))
        if nonlinearity:
            out.add_module("nonlinearity", nonlinearity)

        return out

    def resnet_block(
        self,
        in_filters,
        out_filters,
        size,
        stride,
        padding,
        bias,
        norm_layer,
        nonlinearity,
    ):
        out = nn.Sequential()
        if nonlinearity:
            out.add_module("nonlinearity_0", nonlinearity)
        out.add_module(
            "conv_0",
            nn.Conv2d(
                in_channels=in_filters,
                out_channels=out_filters,
                kernel_size=size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
        )
        if norm_layer:
            out.add_module("normalization", norm_layer(num_features=out_filters))
        if nonlinearity:
            out.add_module("nonlinearity_1", nonlinearity)
        out.add_module(
            "conv_1",
            nn.Conv2d(
                in_channels=in_filters,
                out_channels=out_filters,
                kernel_size=size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
        )
        return out

    def upconv_layer(
        self,
        in_filters,
        out_filters,
        size,
        stride,
        padding,
        bias,
        norm_layer,
        nonlinearity,
    ):
        out = nn.Sequential()
        out.add_module(
            "upconv",
            nn.ConvTranspose2d(
                in_channels=in_filters,
                out_channels=out_filters,
                kernel_size=size,  # 4
                stride=stride,  # 2
                padding=padding,
                bias=bias,
            ),
        )
        if norm_layer:
            out.add_module("normalization", norm_layer(num_features=out_filters))
        if nonlinearity:
            out.add_module("nonlinearity", nonlinearity)
        return out

    def upconv_layer_upsample_and_conv(
        self,
        in_filters,
        out_filters,
        size,
        stride,
        padding,
        bias,
        norm_layer,
        nonlinearity,
    ):
        parts = [
            UpsamplingLayer(in_filters),
            nn.Conv2d(in_filters, out_filters, 3, 1, 1, bias=False),
        ]

        if norm_layer:
            parts.append(norm_layer(num_features=out_filters))

        if nonlinearity:
            parts.append(nonlinearity)

        return nn.Sequential(*parts)


#####
# Default discriminator
#####
class DiscriminatorN_IN(nn.Module):
    def __init__(
        self,
        num_filters=64,
        input_channels=3,
        n_layers=3,
        use_noise=False,
        noise_sigma=0.2,
        norm_layer="instance_norm",
        use_bias=True,
        use_spectral_norm=False,
        use_self_attention=False,
    ):
        super(DiscriminatorN_IN, self).__init__()

        self.num_filters = num_filters
        self.use_noise = use_noise
        self.noise_sigma = noise_sigma
        self.input_channels = input_channels
        self.use_bias = use_bias
        self.use_spectral_norm = use_spectral_norm
        self.use_self_attention = use_self_attention

        if norm_layer == "batch_norm":
            self.norm_layer = nn.BatchNorm2d
        else:
            self.norm_layer = nn.InstanceNorm2d
        self.net = self.make_net(n_layers, self.input_channels, 1, 4, 2, self.use_bias)

        if self.use_self_attention:
            self.self_attention = SelfAttention(
                self.num_filters * 4
            )  # Add after 2nd or 3rd conv layer

    def make_net(self, n, flt_in, flt_out=1, k=4, stride=2, bias=True):
        padding = 1
        model = nn.Sequential()

        model.add_module(
            "conv0",
            self.make_block(
                flt_in, self.num_filters, k, stride, padding, bias, None, nn.LeakyReLU
            ),
        )

        flt_mult, flt_mult_prev = 1, 1
        for l in range(1, n):
            flt_mult_prev = flt_mult
            flt_mult = min(2 ** (l), 8)
            model.add_module(
                "conv_%d" % (l),
                self.make_block(
                    self.num_filters * flt_mult_prev,
                    self.num_filters * flt_mult,
                    k,
                    stride,
                    padding,
                    bias,
                    self.norm_layer,
                    nn.LeakyReLU,
                ),
            )
            if l == 2 and self.use_self_attention:
                model.add_module("self_attention", self.self_attention)

        flt_mult_prev = flt_mult
        flt_mult = min(2**n, 8)
        model.add_module(
            "conv_%d" % (n),
            self.make_block(
                self.num_filters * flt_mult_prev,
                self.num_filters * flt_mult,
                k,
                1,
                padding,
                bias,
                self.norm_layer,
                nn.LeakyReLU,
            ),
        )
        model.add_module(
            "conv_out",
            self.make_block(
                self.num_filters * flt_mult, 1, k, 1, padding, bias, None, None
            ),
        )
        return model

    def make_block(self, flt_in, flt_out, k, stride, padding, bias, norm, relu):
        conv = nn.Conv2d(flt_in, flt_out, k, stride=stride, padding=padding, bias=bias)
        layers = [("conv", conv)]

        if norm is not None:
            norm_layer = norm(flt_out)
            layers.append(("norm", norm_layer))

        if relu is not None:
            relu_layer = relu(0.2, inplace=True)
            layers.append(("relu", relu_layer))

        # Fuse layers if applicable
        block = nn.Sequential(OrderedDict(layers))
        return block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_noise and self.training:
            noise = torch.randn_like(x) * self.noise_sigma
            x = x + noise

        features = []
        for layer in self.net:
            x = layer(x)
            features.append(x)
        return x, features


# class SelfAttention(nn.Module):
#     def __init__(self, in_channels):
#         super(SelfAttention, self).__init__()
#         self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
#         self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
#         self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
#         self.gamma = nn.Parameter(torch.zeros(1))

#     def forward(self, x):
#         batch_size, C, width, height = x.size()
#         query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
#         key = self.key(x).view(batch_size, -1, width * height)
#         energy = torch.bmm(query, key)
#         attention = torch.softmax(energy, dim=-1)
#         value = self.value(x).view(batch_size, -1, width * height)
#         out = torch.bmm(value, attention.permute(0, 2, 1))
#         out = out.view(batch_size, C, width, height)
#         return self.gamma * out + x


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.flash = hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        )  # Check for FlashAttention support

    def forward(self, x):
        batch_size, C, width, height = x.size()

        # Linear projections
        query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, width * height)
        value = self.value(x).view(batch_size, -1, width * height)

        if self.flash:
            # FlashAttention for efficient computation
            attention = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,  # Add an attention mask if needed
                dropout_p=0.0 if self.training else 0.0,  # No dropout for inference
            )
        else:
            # Fallback to manual implementation
            energy = torch.bmm(query, key)
            attention = torch.softmax(energy, dim=-1)
            if self.training:
                attention = torch.nn.functional.dropout(
                    attention, p=0.1
                )  # Apply dropout during training
            attention = torch.bmm(attention, value)

        # Reshape the output
        out = attention.view(batch_size, C, width, height)
        return self.gamma * out + x


#####
# Perception VGG19 loss
#####
class PerceptualVGG19(nn.Module):
    def __init__(self, feature_layers, use_normalization=True, path=None):
        super(PerceptualVGG19, self).__init__()
        # model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1)
        model.float()
        model.eval()

        self.model = model
        self.feature_layers = feature_layers

        self.mean = torch.FloatTensor([0.485, 0.456, 0.406])
        self.mean_tensor = None

        self.std = torch.FloatTensor([0.229, 0.224, 0.225])
        self.std_tensor = None

        self.use_normalization = use_normalization

        if torch.cuda.is_available():
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()

        for param in self.parameters():
            param.requires_grad = False

    def normalize(self, x):
        if not self.use_normalization:
            return x

        if self.mean_tensor is None:
            self.mean_tensor = Variable(
                self.mean.view(1, 3, 1, 1).expand(x.size()), requires_grad=False
            )
            self.std_tensor = Variable(
                self.std.view(1, 3, 1, 1).expand(x.size()), requires_grad=False
            )

        x = (x + 1) / 2
        return (x - self.mean_tensor) / self.std_tensor

    def run(self, x):
        features = []

        h = x

        for f in range(max(self.feature_layers) + 1):
            h = self.model.features[f](h)
            if f in self.feature_layers:
                not_normed_features = h.clone().view(h.size(0), -1)
                features.append(not_normed_features)

        return torch.cat(features, dim=1)

    def forward(self, x):
        h = self.normalize(x)
        return self.run(h)
