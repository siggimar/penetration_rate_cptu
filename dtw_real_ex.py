import numpy as np
import matplotlib.pyplot as plt
from dtaidistance import dtw
from cptu_rate_dataset import dataset

np.random.seed(seed=1234) # reproduceability of example


def dtw_example( n=5 ):
    '''
    example illustrates how dynamic time warping squishes
    tip resistance curves to fit the reference curve
    to run: comment out line #251 in cpt.py: self.calc_lowess_params()
    '''
    save_name = 'dataset_test.pkl'
    my_dataset = dataset()        
    my_dataset.load_dataset( location=0, read_logfiles=True, from_file=save_name )
    my_dataset.save_to_file( save_name )

    reference = my_dataset.get_reference() #12:5mm/s,  1:60mm/s

    fig, axs = plt.subplots( 1, 2, sharey=True, sharex=True )

    # assume log-files are read
    d_ref = reference.log_file.cpt_data['qt']['d']
    qt_ref = reference.log_file.cpt_data['qt']['value']

    axs[0].plot( qt_ref, d_ref, label=reference.pos_name)
    axs[1].plot( qt_ref, d_ref, label=reference.pos_name)

    k=0

    for sounding in my_dataset.soundings:
        if sounding==reference: continue

        d = sounding.log_file.cpt_data['qt']['d']
        qt = sounding.log_file.cpt_data['qt']['value']

        axs[0].plot( qt, d, label=sounding.pos_name)

        qt_resampled = np.interp( d_ref, d, qt )
        alignment = dtw.warping_path_fast( qt_resampled, qt_ref )
        warped_d = restore_d( alignment, d_ref, d_ref )

        axs[1].plot( qt_resampled, warped_d, label=sounding.pos_name )
        k += 1
        if k>=n: break


    if True:
        axs[0].set_xlim(700,1100)  
        axs[0].set_ylim(13,16)  
    
    axs[0].invert_yaxis()
    axs[1].legend()
    plt.show()


def restore_d( path, d_from, d_to ):
    d_restored = np.copy( d_to )

    for item in path:
        d_restored[ item[0] ] = d_from[ item[1] ]
    return d_restored



if __name__ == '__main__':
    dtw_example()