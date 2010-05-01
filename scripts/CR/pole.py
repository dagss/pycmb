from common import *

t = obs.load_temperature_mutable()
t.plot(title='before')
m = obs.properties.load_mask()
print t.remove_multipoles_inplace(2, m)
print t.remove_multipoles_inplace(2, m)
t.plot(title='after')
