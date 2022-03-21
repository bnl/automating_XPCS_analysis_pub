import torch.nn as nn
import torch.nn.functional as F


class Conv2D_params:
    def __init__(
        self,
        dimn_tensor=[None, None, None, None],
        hidden_layers_list=None,
        ksize=None,
        latent_space_dimn=None,
    ):

        self.batchsize = dimn_tensor[0]
        self.channels = dimn_tensor[1]
        self.nX = dimn_tensor[2]
        self.nY = dimn_tensor[3]
        self.hidden_layers_list = hidden_layers_list
        self.ksize = ksize
        self.latent_space_dimn = latent_space_dimn


class Encoder_2D(nn.Module):
    def __init__(self, dimn_tensor, hidden_layers_list, ksize, latent_space_dimn):

        # Input tensors are ( batchsize , channels , nX , nY )

        super(Encoder_2D, self).__init__()

        batchsize, channels, nX, nY = dimn_tensor

        n_layers = len(hidden_layers_list) - 1

        len_signal_conv_X = nX
        len_signal_conv_Y = nY

        # set up convolutional layers
        self.f_conv = nn.ModuleList(
            [
                nn.Conv2d(
                    hidden_layers_list[i],
                    hidden_layers_list[i + 1],
                    kernel_size=ksize[i],
                    padding=(ksize[i] - 1) // 2,
                )
                for i in range(n_layers)
            ]
        )

        for conv_i in self.f_conv:
            nn.init.xavier_uniform_(conv_i.weight)

        # set up linear outout layer
        self.f_linear_out = nn.Linear(
            len_signal_conv_X * len_signal_conv_Y * hidden_layers_list[-1],
            latent_space_dimn,
        )

        nn.init.xavier_uniform_(self.f_linear_out.weight)

        # Save some network parameters
        self.conv2d_params = Conv2D_params(
            dimn_tensor, hidden_layers_list, ksize, latent_space_dimn
        )

    def forward(self, x):

        for conv_i in self.f_conv:
            x = F.relu(conv_i(x))

        batchsize, features, nX, nY = x.size()
        x = self.f_linear_out(x.reshape(batchsize, 1, features * nX * nY))

        return x


class Decoder_2D(nn.Module):
    def __init__(self, latent_dimn, fc_outputsize, nX, nY, channels_list, ksize):

        # Input tensors are ( batchsize , latent_dimn )

        super(Decoder_2D, self).__init__()

        n_layers = len(channels_list) - 1
        ksize = ksize[::-1]

        self.f_linear_in = nn.Linear(latent_dimn, fc_outputsize)

        nn.init.xavier_uniform_(self.f_linear_in.weight)

        self.f_conv = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    channels_list[i],
                    channels_list[i + 1],
                    kernel_size=ksize[i],
                    padding=(ksize[i] - 1) // 2,
                )
                for i in range(n_layers)
            ]
        )

        for conv_i in self.f_conv:
            nn.init.xavier_uniform_(conv_i.weight)

        self.fc_outputsize = fc_outputsize

        self.channels_list = channels_list

        self.nX = nX
        self.nY = nY
        self.nx_conv = nX
        self.ny_conv = nY

    def forward(self, x):

        x = self.f_linear_in(x).reshape(
            x.size()[0], self.channels_list[0], self.nx_conv, self.ny_conv
        )

        for i, conv_i in enumerate(self.f_conv[:-1]):
            x = conv_i(x)
            x = F.relu(x)

        x = self.f_conv[-1](x)

        return x


def get_decoder2d_fcoutputsize_from_encoder2d_params(
    encoder_hidden_layers_list, ksize, nX, nY
):

    decoder_channels = encoder_hidden_layers_list[-1::-1]

    len_signal_conv_X = nX
    len_signal_conv_Y = nY

    fc_outputsize = len_signal_conv_X * len_signal_conv_Y * decoder_channels[0]

    return fc_outputsize


class AutoEncoder_2D(nn.Module):
    def __init__(self, dimn_tensor, hidden_layers_list, ksize, latent_space_dimn):

        super(AutoEncoder_2D, self).__init__()

        self.encoder = Encoder_2D(
            dimn_tensor, hidden_layers_list, ksize, latent_space_dimn
        )

        fc_outputsize = get_decoder2d_fcoutputsize_from_encoder2d_params(
            self.encoder.conv2d_params.hidden_layers_list,
            self.encoder.conv2d_params.ksize,
            self.encoder.conv2d_params.nX,
            self.encoder.conv2d_params.nY,
        )

        self.decoder = Decoder_2D(
            self.encoder.conv2d_params.latent_space_dimn,
            fc_outputsize,
            dimn_tensor[2],
            dimn_tensor[3],
            self.encoder.conv2d_params.hidden_layers_list[-1::-1],
            self.encoder.conv2d_params.ksize,
        )

    def forward(self, x):

        return self.decoder(self.encoder(x))

    def get_latent_space_coordinates(self, x):

        return self.encoder(x)
