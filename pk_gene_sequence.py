import pickle
from data import get_multitask_experiment

pkl=get_multitask_experiment(name='mydataset', scenario='domain', tasks=12, verbose=True, exception=True,factor='sequence')
with open('sequence.pk','wb') as f:
    pickle.dump(pkl,f)
