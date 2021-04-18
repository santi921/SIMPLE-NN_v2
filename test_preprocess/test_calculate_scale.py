import sys
import os
sys.path.append('../')

#DataGenerator.py -> datagenerator.py in utils
#DataGenerator Class -> Datagenerator


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#os.system('python libsymf_builder.py')

from simple_nn_v2.utils import _make_full_featurelist
from simple_nn_v2.utils.mpiclass import DummyMPI
from simple_nn_v2.init_inputs import  initialize_inputs
from simple_nn_v2.features.preprocessing import _split_data, _calculate_scale

# Setting for test preprocessing 

logfile = sys.stdout
logfile = open('LOG', 'w', 10)
inputs = initialize_inputs('input.yaml', logfile)
comm = DummyMPI()


try:
    print('_calculate_scale test')
    print('Preprecessing before _calc_scale')
    tmp_pickle_train, tmp_pickle_valid = _split_data(inputs)
    # generate full symmetry function vector
    feature_list_train, idx_list_train = \
     _make_full_featurelist(tmp_pickle_train, 'x', inputs['atom_types'], is_ptdata=True)
    feature_list_valid, idx_list_valid = \
     _make_full_featurelist(tmp_pickle_valid, 'x', inputs['atom_types'], is_ptdata=True)
    print('Preprocessing done')
    scale = _calculate_scale(inputs, feature_list_train, logfile, calc_scale=True)
    print('_calculate_scale OK')
    print('SCALE : ',scale)
    print('')
except:
    print('!!  Error occured _calculate_scale ')
    print(sys.exc_info())
    print('')
