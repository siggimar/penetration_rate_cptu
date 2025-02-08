from tqdm import tqdm
from cptu_rate_dataset import dataset
from depth_adjuster import depth_adjuster
from cptu_units import unit_model
from cptu_rate_plotter import rate_plotter

from cptu_rate_processor import rate_processor

if __name__=='__main__':
    result_f_name = 'results_unwarped.csv'

    datasets = []
    unit_models = []
    d_adjusters = []

    if False: # mine data, set up models, calculate & consolidate results to a csv
        for location in [0,1,2]:            
            save_name = 'dataset_test.pkl'
            #save_name = '' # read all files again

            datasets.append( dataset() )
            datasets[-1].load_dataset( location=location, read_logfiles=True, from_file=save_name, txt_overview=False ) # location= 0:Tiller-Flotten, 1:Øysand, 2:Halsen-Stjørdal
            #save_name = 'dataset_test.pkl'
            datasets[-1].save_to_file( 'dataset_test.pkl' )

            reference = my_dataset.soundings[6] if False else datasets[-1].get_reference() #Hals07 vs Hals05
            
            
            some_sounding = datasets[0].soundings[11]
            d = some_sounding.log_file.cpt_data['qc']['d']
            import numpy as np
            interval = np.diff(d)
            print(np.average(interval))
            print(np.std(interval))
            a=1
            #datasets[-1].rates_freqs()
            #reference.data_out()
            #reference.plot( plot_log_data=True, marker='o', fontsize=26, tick_f_size=22 )
            #reference.plot()
            #reference.soil_stress_model.plot()
            #reference.plot() # plots data from both CPTu and LOG file (if present)

            unit_models.append( unit_model( datasets[-1] ) )
            d_adjusters.append( depth_adjuster( datasets[-1], unit_models[-1] ) )
            d_adjusters[-1].set_warps() # links warp definition to each cpt

            params = ['qc','qt','fs','u','du','ft','qn','qe','du_sig_v0_eff','Qt_','Bq_','Fr_','Rf_']
            for i in tqdm(range(len(params))):
                param = params[i]
                processor = rate_processor( datasets[-1], unit_models[-1] )
                processor.analize_units( parameter=param )
            processor.calculate_reference_values( visualize=False )

        all_results = processor.consolidate_results( unit_models ) # sort all results pr. unit and sounding
        processor.write_to_csv( result_f_name, all_results )

    else: # generate unit plot
        for location in [0,1,2]:#[2]:##location = 1
            save_name = 'dataset_test.pkl'

            datasets.append( dataset() )
            datasets[-1].load_dataset( location=location, read_logfiles=True, from_file=save_name, txt_overview=False ) # location= 0:Tiller-Flotten, 1:Øysand, 2:Halsen-Stjørdal
            save_name = 'dataset_test.pkl'
            #datasets[-1].save_to_file( 'dataset_test.pkl' )

            #reference = my_dataset.soundings[6] if False else datasets[-1].get_reference()
            unit_models.append( unit_model( datasets[-1] ) )
            d_adjusters.append( depth_adjuster( datasets[-1], unit_models[-1] ) )
            d_adjusters[-1].set_warps() # links warp definition to each cpt
            
            f_scale = 1.0
            d_adjusters[-1].plot_all( fontsize=16*f_scale, tick_f_size=14*f_scale, apply_drops=True, warped=True, draw_units=True )
        
        d = 0 # penetrated meters:  interval x n_soundings
        for um in unit_models:
            for unit in um.units:
                delta_d = ( unit.d_bottom[0]-unit.d_top[0] ) * len( unit.d_top )
                print(unit.unit_name, delta_d)
                d += delta_d

        print('total: ', d)


        unit_models[-1].plot( interpolate=False, plt_boreholes=True, plt_units=True )

    a=1 # here I have unit models with bunch of data ready to be analized