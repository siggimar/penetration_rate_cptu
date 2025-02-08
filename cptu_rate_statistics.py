import os
import pickle
from tqdm import tqdm
import cpt
import soil_stresses
import matplotlib.pyplot as plt
import numpy as np

root = 'C:\\GuDb_data'
save_path = 'rate_statistics.pkl'

# script written to parse through .cpt files found in archived GUDB data
# in all >2100 tests were used to peoduce a statistical view of observed
# penetration rates.  (required SGF file format - endings .cpt or -std)

# to run this script, a copy of gudb archive was used,  see: DOI: 10.5281/zenodo.14837967

def target_file_extension( root, some_extension ):
    ext = some_extension.lower()
    files = [] # reset
    # r=root, d=directories, f = files
    for r, _, f in os.walk( root ):
        for file in f:
            _, file_extension = os.path.splitext( file )
            if ext in file_extension.lower():
                files.append( os.path.join( r, file ) )
    return files


def combine_files_by_ext( root, extensions ):
    files = []
    for ext in extensions:
        files += target_file_extension( root, ext )
    return files


def files_with_cpt_data( candidates ):
    files = []
    used_ext = []
    errors=0
    for i in tqdm(range(len(candidates))):
        cand = candidates[i]
        f_name, f_ext = os.path.splitext( cand )

        has_cpt_data = False
        with open( cand, 'rb' ) as file:
            tmp_bytes = file.read()
        try:
            tmp_cont = tmp_bytes.decode('utf8')

            # check for sgf header
            if 'hm=07' in tmp_cont.lower(): has_cpt_data=True
            if 'hm=7' in tmp_cont.lower(): has_cpt_data=True
            if has_cpt_data: 
                if f_ext.lower() not in used_ext:
                    used_ext.append( f_ext.lower() )
                files.append( cand )
        except Exception as e:
            pass # just ignore errors:  667 encountered in 17432 files
    return files


def calc_stats( cpts, parameter, x_rng=[None,None,None], unit='mm/s', x_label='Penetration rate' ):    
    if x_rng[0] is None: x_rng[0] = 0
    if x_rng[1] is None: x_rng[1] = 50
    if x_rng[2] is None: x_rng[2] = 5
    axis_label_size = 12
    annotation_labels_size = 11
    tick_label_size = 11

    all_rates = []
    all_depths = []
    stop_depths = []
    avg_rates = np.array( [] )
    std_rates = np.array( [] )

    combined_rates = np.array( [] )

    for i in tqdm(range(len(cpts))):
        cpt = cpts[i]
        if parameter in cpt.data:
            tmp_filter = np.isnan(cpt.data[parameter]) # filter out nans - important for fs regs
            tmp_param = cpt.data[parameter][~tmp_filter]
            tmp_d = cpt.data['d'][~tmp_filter]
            
            all_rates.append( tmp_param )
            stop_depths.append( np.max(tmp_d) )
            all_depths.append( tmp_d )
            combined_rates = np.append( combined_rates, tmp_param )
            avg_rates = np.append( avg_rates, np.average(tmp_param) )
            std_rates = np.append( std_rates, np.std(tmp_param) )

    d_min, d_max = 0, 40
    n_min, n_max = 0, 600000
    n_inc = int( (n_max-n_min)/10 )
    r_min, r_max = x_rng[0], x_rng[1]


    fig, axs = plt.subplots(1, 2, figsize=(12,5), gridspec_kw={'width_ratios': [2, 3]})
    plt.subplots_adjust( wspace=0.4 )

    # first ax shows rates against depth
    k=0
    d_sum = 0
    d_max_data = 0
    starts = np.array([])
    ends = np.array([])

    for r, d in zip( all_rates, all_depths ):
        axs[0].plot( r, d, lw=0.005, c=(0/255,142/255,194/255) )
        d_sum += d[-1]-d[0]
        d_max_data = max(d_max_data, d[-1])
        starts = np.append(starts, d[0])
        ends = np.append(ends, d[-1])
        k+=1


    idx = np.arange( 0 , 101, 1 )
    #all_percentiles = np.array( [ np.percentile(combined_rates, i) for i in idx ] )
    all_percentiles = np.array([ np.percentile(avg_rates, i) for i in idx ])
    p_depths = np.array([ np.percentile(stop_depths, i) for i in idx ])

    for i, val in enumerate(p_depths):
        pass#print(i,val)
        
    # print some statistics for the presented data
    if True:
        p_15_25 = np.where(all_percentiles > 25)[0][0] - np.where(all_percentiles < 15)[0][-1]
        p_19_21 = np.where(all_percentiles > 21)[0][0] - np.where(all_percentiles < 19)[0][-1]
        p_10_30 = np.where(all_percentiles > 30)[0][0] - np.where(all_percentiles < 10)[0][-1]



        print('p:10-30', p_10_30)
        print('p:15-25', p_15_25)
        print('p:19-21', p_19_21)

        print( 'k:', k )
        print( 'D:', d_sum)
        print( 'std:', np.std(combined_rates))
        print( 'avgstd:', np.average(std_rates))
        print( 'avgavg:', np.average(avg_rates))
        print( 'avg_all:', np.average(combined_rates))
        print( 'max_d:', d_max_data)

        print( 'avg_start_d:', np.average(starts))
        print( 'avg_end_d:', np.average(ends))
        print( 'med_end_d:', np.median(ends))

    axs[0].plot( r*2*r_max, d, lw=1, c=(0/255,142/255,194/255), label='test data - ' + str(round(k-5,-1)) + ' tests' ) # for legend - outside plot

    # second ax is a histogram
    bin_width = round((x_rng[1]+1-x_rng[0])/50,1)
    bin_range = (x_rng[0]-.5, x_rng[1])
    axs[1].hist( combined_rates, bins=np.arange(bin_range[0], bin_range[1] + bin_width, bin_width), ec='k', fc=(0/255,142/255,194/255), label='Test data - ' + str(round(k-5,-1)) + ' tests' ) #, density=True )

    perc_vals = [ [25,75], [10,90] ]
    perc = [ [ np.percentile( combined_rates, perc_vals[0][0]),np.percentile( combined_rates, perc_vals[0][1])],
             [ np.percentile( combined_rates, perc_vals[1][0]),np.percentile( combined_rates, perc_vals[1][1])]]

    perc_colors = [(255/255,150/255,0/255),(237/255,28/255,46/255) ] # NPRA orange/red
    lws = [2,2]
    zorders = [2, -2]

    for j in range(2):
        i=0
        if parameter=='b': axs[j].plot( [20, 20], [n_min, n_max], lw=lws[j], ls='--', c=(0,0,0), label='Standard rate', zorder=3)        

        for p, vals in zip( perc, perc_vals ):
            label = str(vals[0]) + '/' + str(vals[1]) + 'th percentile'

            axs[j].plot( [p[0], p[0]], [n_min, n_max], lw=lws[j], ls='--', c=perc_colors[i], label=label, zorder=zorders[j] )
            axs[j].plot( [p[1], p[1]], [n_min, n_max], lw=lws[j], ls='--', c=perc_colors[i], zorder=zorders[j] )
            i += 1

    axs[0].plot( r, d, lw=0.02, c=(0/255,142/255,194/255) )
    axs[0].set_xlim( r_min, r_max )
    axs[0].set_ylim( d_min, d_max )    
    axs[0].invert_yaxis()

    x_min, x_max = axs[1].get_xlim()
    x = np.linspace(x_min,x_max,500)
    axs[1].fill_between( x, 0, 1, where= np.logical_and(x>perc[1][0],x<perc[0][0]), color=perc_colors[1], alpha=0.3, transform=axs[1].get_xaxis_transform(),zorder=-1)
    axs[1].fill_between( x, 0, 1, where= np.logical_and(x>perc[0][1],x<perc[1][1]), color=perc_colors[1], alpha=0.3, transform=axs[1].get_xaxis_transform(),zorder=-1)
    axs[1].fill_between( x, 0, 1, where= np.logical_and(x>perc[0][0],x<perc[0][1]), color=perc_colors[0], alpha=0.3, transform=axs[1].get_xaxis_transform(),zorder=-1)

    axs[1].set_ylim( n_min, n_max ) # 0.25 for pdf variant

    # axis labels    
    axs[0].set_xlabel(x_label + ' (' + unit + ')', fontsize=axis_label_size )
    axs[0].set_ylabel('Depth (m)', fontsize=axis_label_size)
    axs[1].set_xlabel(x_label + ' (' + unit + ')', fontsize=axis_label_size )
    axs[1].set_ylabel('Number of ' + x_label.lower() + ' registrations (-)', fontsize=axis_label_size)

    # annotate
    if parameter=='b':
        axs[1].annotate('Push breaks', xy=(0.2,7000), xycoords='data',
        xytext=(0,75000), textcoords='data', va='top', ha='left', fontsize=annotation_labels_size,
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

        axs[1].annotate('CPTUs at total-\nsounding rates', xy=(47,7000), xycoords='data',
        xytext=(36,105000), textcoords='data', va='top', ha='left', fontsize=annotation_labels_size,
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    axs[1].annotate(str(perc_vals[1][0]) + 'th: ' + str(round(perc[1][0],1)) + unit, xy=(perc[1][0],320000), xycoords='data',
    xytext=(2,330000), textcoords='data', va='top', ha='left', fontsize=annotation_labels_size,
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    axs[1].annotate(str(perc_vals[0][0]) + 'th: ' + str(round(perc[0][0],1)) + unit, xy=(perc[0][0],420000), xycoords='data',
    xytext=(2,430000), textcoords='data', va='top', ha='left', fontsize=annotation_labels_size,
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    axs[1].annotate('50th: ' + str(round(np.percentile( combined_rates, 50),1)) + unit, xy=(np.percentile( combined_rates, 50),530000), xycoords='data',
    xytext=(2,540000), textcoords='data', va='top', ha='left', fontsize=annotation_labels_size,
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    axs[1].annotate(str(perc_vals[0][1]) + 'th: ' + str(round(perc[0][1],1)) + unit, xy=(perc[0][1],420000-n_inc), xycoords='data',
    xytext=(round(perc[1][1],1)+3*bin_width,430000-n_inc), textcoords='data', va='top', ha='left', fontsize=annotation_labels_size,
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    axs[1].annotate(str(perc_vals[1][1]) + 'th: ' + str(round(perc[1][1],1)) + unit, xy=(perc[1][1],320000-n_inc), xycoords='data',
    xytext=(round(perc[1][1],1)+3*bin_width,330000-n_inc), textcoords='data', va='top', ha='left', fontsize=annotation_labels_size,
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    #turn off top and right spine
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)

    # adjust ticks
    axs[0].tick_params( labelsize=tick_label_size )
    axs[1].tick_params( labelsize=tick_label_size )

    axs[0].yaxis.set_ticks( np.arange( 0, 41, 10 ) )
    axs[1].xaxis.set_ticks( np.arange( x_rng[0], x_rng[1]+1, x_rng[2] ) )

    axs[0].legend(loc='lower right')
    axs[1].legend(loc='upper right')

    axs[0].text( 2, 2.5, 'A)', size=16, bbox=dict(boxstyle="square",ec=(1, 1, 1,.7),fc=(1, 1, 1,.7)))
    axs[1].text(-2, 570000, 'B)', size=16, bbox=dict(boxstyle="square",ec=(1, 1, 1),fc=(1, 1, 1)))
    
    
    plt.tight_layout(w_pad=2.0)
    plt.show()


# saving/loading from file saves minutes on each run
def save_to_file( some_var ):
    with open( save_path, 'wb' ) as file:
        pickle.dump( some_var, file )

def load_from_file( ):
    with open( save_path, 'rb') as file:
        some_var = pickle.load( file )
    return some_var


if __name__=='__main__':
    if os.path.isfile( save_path ):
        cpts = load_from_file()
    else:
        # list files
        extensions = [ '.cpt', '.std' ]
        candidates = combine_files_by_ext( root=root, extensions=extensions )
        cpt_files = files_with_cpt_data( candidates )
        # bogus stress mdoel
        sm = soil_stresses.soil_stress_model( gamma=[[0],[19]], u0=[[0,1,100],[0,0,990]])

        # load all cpt data into classes ( 2nd tqdm )
        cpts = []
        for i in tqdm(range(len(cpt_files))):            
            some_cpt_file = cpt_files[i]
            some_cpt = cpt.cpt(some_cpt_file,soil_stress_model=sm, read_log_file=False, ignore_calibration=True)
            if some_cpt.remove_sounding: continue
            cpts.append( some_cpt )
        save_to_file( cpts ) # 2429 tests

    calc_stats( cpts, parameter='b' )#'u', x_rng=[0,2000,500], unit='kPa', x_label='u' + r'$_2$' )