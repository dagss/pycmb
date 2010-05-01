from common import *
from matplotlib import pyplot as plt

with working_directory('/mn/corcaroli/d1/dagss/commander'):
    if os.path.exists('cr_beam.fits'):
        os.unlink('cr_beam.fits')
    beam_to_fits('cr_beam.fits', beam)
    obs.load_temperature().to_fits('cr_temp.fits')
    obsprop.load_mask().to_fits('cr_mask.fits', 'raw')
    obsprop.load_rms('nested').to_fits('cr_rms.fits')

#model.plot(lmax=70, scale=False)

#Cl = model.get_power_spectrum()
#plt.plot(Cl)
