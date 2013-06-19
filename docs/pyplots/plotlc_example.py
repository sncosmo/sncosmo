import numpy as np
import sncosmo

model = sncosmo.get_model('salt2')
model.set(z=0.5, c=0.2, t0=55100., mabs=-19.5, x1=0.5)

times = 55070. + 2. * np.arange(30, dtype=np.float)
bands = np.array(10 * ['desg', 'desr', 'desi'])
zp = 25. * np.ones(30)
zpsys = np.array(30 * ['ab'])

flux = model.bandflux(bands, times, zp=zp, zpsys=zpsys)
fluxerr = (0.05 * np.max(flux)) * np.ones(30, dtype=np.float)
flux += fluxerr * np.random.randn(30)

data = {'time': times, 'band': bands, 'flux': flux, 
        'fluxerr': fluxerr, 'zp': zp, 'zpsys': zpsys}

sncosmo.plotlc(data, model=model)
