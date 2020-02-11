import pickle
from data import get_multitask_experiment
factors=['clutter','illumination','occlusion','pixel']

for factor in factors:
    pkl=get_multitask_experiment(name='mydataset', scenario='domain', tasks=9, verbose=True, exception=True,factor=factor)
    with open(factor+'.pk','wb') as f:
        pickle.dump(pkl,f)
'''
pkl=get_multitask_experiment(name='mydataset', scenario='domain', tasks=36, verbose=True, exception=True,factor='sequence')
with open('sequence.pk','wb') as f:
    pickle.dump(pkl,f)
'''