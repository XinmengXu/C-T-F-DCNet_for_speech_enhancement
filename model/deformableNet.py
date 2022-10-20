import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from model.deformable.deform_conv import th_batch_map_offsets, th_generate_grid
from model.DCCRN.ConvSTFT import ConvSTFT, ConviSTFT 



class C_T_F_Attention(nn.Module):
    def __init__(self, channel = 256, time = 251, frequency = 7, num_heads=1):
        super(C_T_F_Attention, self).__init__()
        self.num_heads = num_heads
##################################################################################################################
        self.q_conv_c = nn.Conv1d(time, time, kernel_size=1, bias=False)
        self.q_bn_c = nn.BatchNorm1d(time)
		
        self.k_conv_c = nn.Conv1d(time, time, kernel_size=1, bias=False)
        self.k_bn_c = nn.BatchNorm1d(time)
		
        self.logits_bn_c = nn.BatchNorm2d(1)
##################################################################################################################
        self.q_conv_f = nn.Conv1d(channel, channel, kernel_size=1, bias=False)
        self.q_bn_f = nn.BatchNorm1d(channel)

        self.k_conv_f = nn.Conv1d(channel, channel, kernel_size=1, bias=False)
        self.k_bn_f = nn.BatchNorm1d(channel)

        self.logits_bn_f = nn.BatchNorm2d(1)
##################################################################################################################
        self.q_conv_t = nn.Conv1d(channel, channel, kernel_size=1, bias=False)
        self.q_bn_t = nn.BatchNorm1d(channel)

        self.k_conv_t = nn.Conv1d(channel, channel, kernel_size=1, bias=False)
        self.k_bn_t = nn.BatchNorm1d(channel)

        self.logits_bn_t = nn.BatchNorm2d(1)
##################################################################################################################
        self.v_conv = nn.Conv2d(channel, channel, kernel_size=(1, 1))
        self.v_bn = nn.BatchNorm2d(channel)
		
		
    def forward(self, x):

        x_c = x.permute(0, 2, 3, 1)  # batch_size, width, depth, height
        #print(x_c.size())
        x_f = x.permute(0, 2, 1, 3)
        #print(x_f.size())
        x_t = x.permute(0, 3, 1, 2)        
        #print(x_t.size())		


#########################################for channel#######################################################
            
        batch_size, width, depth, height = x_c.size()# 1 x 7 x 251 x 256
        x_c = x_c.reshape(batch_size * width, depth, height) # 7 x 251 x 256

        q_c = self.q_bn_c(self.q_conv_c(x_c))
        k_c = self.k_bn_c(self.k_conv_c(x_c))

        k_c = k_c.unsqueeze(1)
        q_c = q_c.unsqueeze(1)


        k_c = F.adaptive_avg_pool2d(k_c, (1, 256)).squeeze(-1)
        q_c = F.adaptive_avg_pool2d(q_c, (1, 256)).squeeze(-1)	
    

        qk_c = torch.matmul(q_c.transpose(2, 3), k_c)

        logits_c = self.logits_bn_c(qk_c) # apply batch normalization on qk, qr, kr
        
        weights_c = F.softmax(logits_c, dim=3) # 7 x 1 x 256 x 256

###########################################for frequency#####################################################
        batch_size, width, depth, height = x_f.size() # 1 x 7 x 256 x 251
        x_f = x_f.reshape(batch_size * width, depth, height) # 7 x 256 x 251

        q_f = self.q_bn_f(self.q_conv_f(x_f)) # 7 x 256 x 251
        k_f = self.k_bn_f(self.k_conv_f(x_f)) # 7 x 256 x 251

        k_f = k_f.unsqueeze(1) # 7 x 1 x 256 x 251
        q_f = q_f.unsqueeze(1) # 7 x 1 x 256 x 251

        k_f = F.adaptive_avg_pool2d(k_f, (1, 251)).squeeze(-1) # 7 x 1 x 1 x 251
        q_f = F.adaptive_avg_pool2d(q_f, (1, 251)).squeeze(-1) # 7 x 1 x 1 x 251

        qk_f = torch.matmul(q_f.transpose(2, 3), k_f)  # 7 x 1 x 251 x 251

        logits_f = self.logits_bn_f(qk_f) # 7 x 1 x 251 x 251

        weights_f = F.softmax(logits_f, dim=3) # 


#############################################for time########################################################
        batch_size, width, depth, height = x_t.size() # 1 x 251 x 256 x 7
        x_t = x_t.reshape(batch_size * width, depth, height) # 251 x 256 x 7

        q_t = self.q_bn_t(self.q_conv_t(x_t)) # 251 x 256 x 7
        k_t = self.k_bn_t(self.k_conv_t(x_t)) # 251 x 256 x 7

        k_t = k_t.unsqueeze(1) # 251 x 1 x 256 x 7
        q_t = q_t.unsqueeze(1) # 251 x 1 x 256 x 7

        k_t = F.adaptive_avg_pool2d(k_t, (1, 7)).squeeze(-1) # 251 x 1 x 1 x 7
        q_t = F.adaptive_avg_pool2d(q_t, (1, 7)).squeeze(-1) # 251 x 1 x 1 x 7	

        qk_t = torch.matmul(q_t.transpose(2, 3), k_t)  # 251 x 1 x 7 x 7

        logits_t = self.logits_bn_t(qk_t) # 251 x 1 x 7 x 7

        weights_t = F.softmax(logits_t, dim=3)

##################################################################################################################
        v = self.v_bn(self.v_conv(x)) # 1 x 256 x 7 x 251
        #print(v.size())
        weights_c = weights_c.squeeze(1)
        #print(weights_c.size())
        weights_f = weights_f.squeeze(1)
        #print(weights_f.size())
        weights_t = weights_t.squeeze(1)
        #print(weights_t.size())
		
        v_c = torch.einsum('bhxy,ihh->bhxy', v, weights_c)
        v_f = torch.einsum('bhxy,iyy->bhxy', v, weights_f)
        v_t = torch.einsum('bhxy,jxx->bhxy', v, weights_t)		
        output = v_t + v_c + v_f + x

        return output


class ConvOffset2D(nn.Conv2d):
    """ConvOffset2D
    Convolutional layer responsible for learning the 2D offsets and output the
    deformed feature map using bilinear interpolation
    Note that this layer does not perform convolution on the deformed feature
    map. See get_deform_cnn in cnn.py for usage
    """
    def __init__(self, filters, init_normal_stddev=0.01, **kwargs):
        """Init
        Parameters
        ----------
        filters : int
            Number of channel of the input feature map
        init_normal_stddev : float
            Normal kernel initialization
        **kwargs:
            Pass to superclass. See Con2d layer in pytorch
        """
        self.filters = filters
        self._grid_param = None
        super(ConvOffset2D, self).__init__(self.filters, self.filters*2, 3, padding=1, bias=False, **kwargs)
        self.weight.data.copy_(self._init_weights(self.weight, init_normal_stddev))

    def forward(self, x):
        """Return the deformed featured map"""
        x_shape = x.size()
        offsets = super(ConvOffset2D, self).forward(x)
        #print(offsets.size())
        # offsets: (b*c, h, w, 2)
        offsets = self._to_bc_h_w_2(offsets, x_shape)
        # x: (b*c, h, w)
        x = self._to_bc_h_w(x, x_shape)
        # X_offset: (b*c, h, w)
        x_offset = th_batch_map_offsets(x, offsets, grid=self._get_grid(self,x))
        # x_offset: (b, h, w, c)
        x_offset = self._to_b_c_h_w(x_offset, x_shape)
        return x_offset

    @staticmethod
    def _get_grid(self, x):
        batch_size, input_height, input_width = x.size(0), x.size(1), x.size(2)
        dtype, cuda = x.data.type(), x.data.is_cuda
        if self._grid_param == (batch_size, input_height, input_width, dtype, cuda):
            return self._grid
        self._grid_param = (batch_size, input_height, input_width, dtype, cuda)
        self._grid = th_generate_grid(batch_size, input_height, input_width, dtype, cuda)
        return self._grid

    @staticmethod
    def _init_weights(weights, std):
        fan_out = weights.size(0)
        fan_in = weights.size(1) * weights.size(2) * weights.size(3)
        w = np.random.normal(0.0, std, (fan_out, fan_in))
        return torch.from_numpy(w.reshape(weights.size()))

    @staticmethod
    def _to_bc_h_w_2(x, x_shape):
        """(b, 2c, h, w) -> (b*c, h, w, 2)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]), 2)
        return x

    @staticmethod
    def _to_bc_h_w(x, x_shape):
        """(b, c, h, w) -> (b*c, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))
        return x

    @staticmethod
    def _to_b_c_h_w(x, x_shape):
        """(b*c, h, w) -> (b, c, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[1]), int(x_shape[2]), int(x_shape[3]))
        return x

class GLayerNorm2d(nn.Module):
    
    def __init__(self, in_channel, eps=1e-12):
        super(GLayerNorm2d, self).__init__()
        self.eps = eps 
        self.beta = nn.Parameter(torch.ones([1, in_channel,1,1]))
        self.gamma = nn.Parameter(torch.zeros([1, in_channel,1,1]))
    
    def forward(self,inputs):
        mean = torch.mean(inputs,[1,2,3], keepdim=True)
        var = torch.var(inputs,[1,2,3], keepdim=True)
        outputs = (inputs - mean)/ torch.sqrt(var+self.eps)*self.beta+self.gamma
        return outputs
		

class NonCausalConvBlock1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 2),
            stride=(2, 1),
            padding=(0, 1)
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]
        Returns:
            [B, C, F, T]
        """
        x = self.conv(x)
        x = x[:, :, :, :-1]  # chomp size
        x = self.norm(x)
        x = self.activation(x)
        return x

class NonCausalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.offset = ConvOffset2D(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 2),
            stride=(2, 1),
            padding=(0, 1)
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 2),
            stride=(2, 1),
            padding=(0, 1)
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]
        Returns:
            [B, C, F, T]
        """
        x1 = self.conv1(x)
        x2 = self.sigmoid(self.conv2(self.offset(x)))
        x = x1 * x2       
        x = x1[:, :, :, :-1]  # chomp size
        x = self.norm(x)
        x = self.activation(x)
        return x

class NonCausalTransConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_last=False, output_padding=(0, 0)):
        super().__init__()
        self.offset = ConvOffset2D(in_channels)
        self.conv1 = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 2),
            stride=(2, 1),
            output_padding=output_padding
        )
        self.conv2 = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 2),
            stride=(2, 1),
            output_padding=output_padding
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        if is_last:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]
        Returns:
            [B, C, F, T]
        """
        x1 = self.conv1(x)
        x2 = self.sigmoid(self.conv2(self.offset(x)))
        x = x1 * x2       
        x = x1[:, :, :, :-1]  # chomp size
        x = self.norm(x)
        x = self.activation(x)
        return x

class NonCausalTransConvBlock1(nn.Module):
    def __init__(self, in_channels, out_channels, is_last=False, output_padding=(0, 0)):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 2),
            stride=(2, 1),
            output_padding=output_padding
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        if is_last:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.ReLU()

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]
        Returns:
            [B, C, F, T]
        """
        x = self.conv(x)
        x = x[:, :, :, :-1]  # chomp size
        x = self.norm(x)
        x = self.activation(x)
        return x
		
class FeedForwardNetwork(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=dim,
                               out_channels=dim,
                               kernel_size=1,
                               stride=1)
        self.conv2 = nn.Conv2d(in_channels=dim,
                               out_channels=dim,
                               kernel_size=1,
                               stride=1)
        self.conv3 = nn.Conv2d(in_channels=dim,
                               out_channels=dim,
                               kernel_size=1,
                               stride=1)

        self.layer1 = nn.Linear(251, 500)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.layer2 = nn.Linear(251, 251)

    def forward(self, x):
        out1 = self.tanh(self.conv1(x))
        out2 = self.sigmoid(self.conv2(x))
        x = out1 * out2
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

		
class Model(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self, channel_amp = 1, channel_phase=2):
        super(Model, self).__init__()
        self.stft = ConvSTFT(512, 256, 512, 'hanning', 'complex', True)
        self.istft = ConviSTFT(512, 256, 512, 'hanning', 'complex', True)
		
        self.amp_conv1 = nn.Sequential(
                                nn.Conv2d(2, channel_amp, 
                                        kernel_size=[7,1],
                                        padding=(3,0)
                                    ),
                                nn.BatchNorm2d(channel_amp),
                                nn.ReLU(),
                                nn.Conv2d(channel_amp, channel_amp, 
                                        kernel_size=[1,7],
                                        padding=(0,3)
                                    ),
                                nn.BatchNorm2d(channel_amp),
                                nn.ReLU(),
                        )
        self.phase_conv1 = nn.Sequential(
                                nn.Conv2d(3, channel_phase, 
                                        kernel_size=[3,5],
                                        padding=(1,2)
                                    ),
                                nn.Conv2d(channel_phase, channel_phase, 
                                        kernel_size=[3,25],
                                        padding=(1, 12)
                                    ),
                        )
        self.amp_conv2 = nn.Sequential(
                        nn.Conv2d(channel_amp, 1, kernel_size=[1, 1]),
                        nn.BatchNorm2d(1),
                        nn.ReLU(),
                    )
        self.phase_conv2 = nn.Sequential(
                        nn.Conv1d(2,8,kernel_size=[1,1])
                    )		
        # Encoder
        self.conv_block_1 = NonCausalConvBlock1(1, 16)
        self.conv_block_2 = NonCausalConvBlock(16, 32)
        self.conv_block_3 = NonCausalConvBlock(32, 64)
        self.conv_block_4 = NonCausalConvBlock(64, 128)
        self.conv_block_5 = NonCausalConvBlock(128, 256)
        self.lstm_layer = nn.LSTM(input_size=251, hidden_size=251, num_layers=2, batch_first=True)
        self.atten = C_T_F_Attention()
        self.norm = nn.LayerNorm([7, 251])

        self.ffn = FeedForwardNetwork(256)		

        # self.SA_time = Axial_Layer(251, height_dim=True)
        # self.SA_frequency = Axial_Layer(256, kernel_size=251, height_dim = False)				 
        self.tran_conv_block_1 = NonCausalTransConvBlock1(256 + 256, 128)
        self.tran_conv_block_2 = NonCausalTransConvBlock1(128 + 128, 64)
        self.tran_conv_block_3 = NonCausalTransConvBlock1(64 + 64, 32)
        self.tran_conv_block_4 = NonCausalTransConvBlock1(32 + 32, 16, output_padding=(1, 0))
        self.tran_conv_block_5 = NonCausalTransConvBlock1(16, 1, is_last=True)
        self.phase_conv5 = nn.Sequential(
                        nn.Conv1d(8, 2, kernel_size=(1,1))
                    )
        self.phase_conv3 = nn.Sequential(
                        nn.Conv2d(8, 8, kernel_size=(5,5), padding=(2,2)),
                        GLayerNorm2d(8),
                    )
        self.phase_conv4 = nn.Sequential(
                        nn.Conv2d(8, 8, kernel_size=(1,25), padding=(0,12)),
                        GLayerNorm2d(8),
                    )

        self.rnn = nn.GRU(
                        257,
                        300,
                        bidirectional=True
                    )
        self.fcs = nn.Sequential(
                    nn.Linear(300*2,600),
                    nn.ReLU(),
                    nn.Linear(600,600),
                    nn.ReLU(),
                    nn.Linear(600,514//2),
                    nn.Sigmoid()
                )
    def forward(self, x):
        #self.lstm_layer.flatten_parameters()
        # [B, D*2, T]
        cmp_spec = self.stft(x)
        cmp_spec = torch.unsqueeze(cmp_spec, 1)

        # to [B, 2, D, T]
        cmp_spec = torch.cat([
                                cmp_spec[:,:,:257,:],
                                cmp_spec[:,:,257:,:],
                                ],
                                1)
        mean = torch.mean(cmp_spec, [1, 2, 3], keepdim = True)
        std = torch.std(cmp_spec, [1, 2, 3], keepdim = True)
        cmp_spec = (cmp_spec - mean) / (std + 1e-8)    
        amp_spec = torch.sqrt(
                            torch.abs(cmp_spec[:,0])**2+
                            torch.abs(cmp_spec[:,1])**2,
                        )
        phase_spec = torch.angle(cmp_spec)
        amp_spec = torch.unsqueeze(amp_spec, 1)
        spec = self.amp_conv1(cmp_spec)
        e_1 = self.conv_block_1(spec)
        e_2 = self.conv_block_2(e_1)
        e_3 = self.conv_block_3(e_2)
        e_4 = self.conv_block_4(e_3)
        e_5 = self.conv_block_5(e_4)  # [2, 256, 4, 200]

        # atten1 = self.norm(self.atten(e_5))# + e_5
        # atten2 = self.norm(self.atten(atten1))# + atten1
        # atten3 = self.norm(self.atten(atten2))# + atten2
		
        atten1 = self.norm(self.atten(e_5))
        atten1 = self.ffn(atten1)
        atten1 = self.norm(atten1)
		
        atten2 = self.norm(self.atten(atten1))
        atten2 = self.ffn(atten2)
        atten2 = self.norm(atten2)

        atten3 = self.norm(self.atten(atten2))
        atten3 = self.ffn(atten3)
        atten3 = self.norm(atten3)
        
		
        # batch_size, n_channels, n_f_bins, n_frame_size = e_5.shape

        # # [2, 256, 4, 200] = [2, 1024, 200] => [2, 200, 1024]
        # lstm_in = e_5.reshape(batch_size, n_channels * n_f_bins, n_frame_size)#.permute(0, 2, 1)
        # lstm_out, _ = self.lstm_layer(lstm_in)  # [2, 200, 1024]
        # lstm_out = lstm_out.reshape(batch_size, n_channels, n_f_bins, n_frame_size)  # [2, 256, 4, 200]
		
        d_1 = self.tran_conv_block_1(torch.cat((atten3, e_5), 1))
        d_2 = self.tran_conv_block_2(torch.cat((d_1, e_4), 1))
        d_3 = self.tran_conv_block_3(torch.cat((d_2, e_3), 1))
        d_4 = self.tran_conv_block_4(torch.cat((d_3, e_2), 1))
        d = self.tran_conv_block_5(d_4)

	
        phase_pro = self.phase_conv1(torch.cat([phase_spec, d], dim = 1))		
        phase_input = torch.cat([phase_pro, self.amp_conv2(d)], dim = 1)
      
        phase_input = self.phase_conv2(phase_pro)	
        p1 = self.phase_conv3(phase_input)
        p1 = self.phase_conv4(p1)
		
        p2 = self.phase_conv3(p1 + phase_input)
        p2 = self.phase_conv4(p2)
		
        p3 = self.phase_conv3(p2 + p1)
        p3 = self.phase_conv4(p3)

        p5 = self.phase_conv5(p3)
        p5 = phase_spec + p5
        p5 = p5/(torch.sqrt(
                            torch.abs(p5[:,0])**2+
                            torch.abs(p5[:,1])**2)
                        +1e-8).unsqueeze(1)
        est_spec = d * p5
        est_spec = torch.cat([est_spec[:,0], est_spec[:,1]], 1)
        est_wav = self.istft(est_spec, None)
        est_wav = torch.squeeze(est_wav, 1)
        return est_spec, est_wav
    def loss(self, est, labels, mode='Mix'):
        '''
        mode == 'Mix'
            est: [B, F*2, T]
            labels: [B, F*2,T]
        mode == 'SiSNR'
            est: [B, T]
            labels: [B, T]
        '''
        if mode == 'SiSNR':
            if labels.dim() == 3:
                labels = torch.squeeze(labels,1)
            if est.dim() == 3:
                est = torch.squeeze(est,1)
            return -si_snr(est, labels)         
        elif mode == 'Mix':
            b, d, t = est.size()
            gth_cspec = self.stft(labels)
            est_cspec = est  
            gth_mag_spec = torch.sqrt(
                                    gth_cspec[:, :self.feat_dim, :]**2
                                    +gth_cspec[:, self.feat_dim:, :]**2 + 1e-8
                               )
            est_mag_spec = torch.sqrt(
                                    est_cspec[:, :self.feat_dim, :]**2
                                    +est_cspec[:, self.feat_dim:, :]**2 + 1e-8
                                )
            
            # power compress 
            gth_cprs_mag_spec = gth_mag_spec**0.3
            est_cprs_mag_spec = est_mag_spec**0.3
            amp_loss = F.mse_loss(
                                gth_cprs_mag_spec, est_cprs_mag_spec
                            )*d
            compress_coff = (gth_cprs_mag_spec/(1e-8+gth_mag_spec)).repeat(1,2,1)
            phase_loss = F.mse_loss(
                                gth_cspec*compress_coff,
                                est_cspec*compress_coff
                            )*d
            
            all_loss = amp_loss*0.5 + phase_loss*0.5
            return all_loss, amp_loss, phase_loss

def remove_dc(data):
    mean = torch.mean(data, -1, keepdim=True) 
    data = data - mean
    return data
def l2_norm(s1, s2):
    #norm = torch.sqrt(torch.sum(s1*s2, 1, keepdim=True))
    #norm = torch.norm(s1*s2, 1, keepdim=True)
    
    norm = torch.sum(s1*s2, -1, keepdim=True)
    return norm 

def si_snr(s1, s2, eps=1e-8):
    #s1 = remove_dc(s1)
    #s2 = remove_dc(s2)
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target =  s1_s2_norm/(s2_s2_norm+eps)*s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10*torch.log10((target_norm)/(noise_norm+eps)+eps)
    return torch.mean(snr)

def test_attention():
    torch.manual_seed(20)
    inputs = torch.randn(1, 64000)
    net = Model()
    a, d = net(inputs)
    print(a.shape)

if __name__ == '__main__':
    test_attention()
