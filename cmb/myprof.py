import cProfile
import sys
import cmb.lsprofcalltree

subject = sys.argv[1]
if subject.endswith('.py'):
    subject = subject[:-3]
p = cProfile.Profile().run('import %s' % subject)
k = cmb.lsprofcalltree.KCacheGrind(p)
with file('%s.kgrind' % subject, 'w') as f:
    k.output(f)

