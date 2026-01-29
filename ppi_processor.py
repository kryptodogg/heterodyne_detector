import torch
import numpy as np

class PPIProcessor:
    def __init__(self, geometry, config, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.d, self.c, self.g = device, config, geometry
        na, nrb, mrm = c['num_angles'], c['num_range_bins'], c['max_range_m']
        self.na, self.nrb, self.mrm, self.bf = na, nrb, mrm, c.get('beamformer', 'conventional')

        self.ad = np.linspace(-90, 90, na)
        self.sv = geometry.compute_steering_vector(torch.tensor(self.ad * np.pi / 180, dtype=torch.float32, device=d)).to(d)
        self.win = torch.hann_window(nrb, periodic=False, dtype=torch.float32, device=d)
        self.map = torch.zeros((na, nrb), dtype=torch.float32, device=d)

    def _mvdr_weights(self, x, sv_rx):
        N, r11, r12, r22 = x.shape[1], torch.sum(x[0]*x[0].conj())/N, torch.sum(x[0]*x[1].conj())/N, torch.sum(x[1]*x[1].conj())/N
        reg, r21, det = 1e-6, r12.conj(), (r11+reg)*(r22+reg) - r12*r21
        inv11, inv12, inv21, inv22 = (r22+reg)/det, -r12/det, -r21/det, (r11+reg)/det
        w1 = inv11 * sv_rx[0] + inv12 * sv_rx[1]
        w2 = inv21 * sv_rx[0] + inv22 * sv_rx[1]
        return w1.conj() * x[0] + w2.conj() * x[1]

    def _range_profile(self, sig):
        nrb = self.nrb
        sig = sig[:nrb] if len(sig) >= nrb else torch.cat([sig, torch.zeros(nrb-len(sig), dtype=sig.dtype, device=sig.device)])
        return 20 * torch.log10(torch.abs(torch.fft.fft(sig * self.win, n=nrb)) + 1e-10)

    def process(self, rx1, rx2):
        rx1, rx2 = rx1.to(self.d), rx2.to(self.d)
        X = torch.stack([rx1, rx2], dim=0)
        sv_rx = self.sv[:, :2].conj()

        if self.bf == 'mvdr':
            [self.map.__setitem__(i, self._range_profile(self._mvdr_weights(X, sv_rx[i]))) for i in range(self.na)]
        else:  # conventional
            bf_all = torch.sum(sv_rx.unsqueeze(-1) * X.unsqueeze(0), dim=1)
            [self.map.__setitem__(i, self._range_profile(bf_all[i])) for i in range(self.na)]

        return {
            'ppi_map': self.map.cpu().numpy(),
            'angles_deg': self.ad,
            'ranges_m': np.linspace(0, self.mrm, self.nrb)
        }