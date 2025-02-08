import numpy as np
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm

# class written to do rate dependency analysis on each unit

class rate_processor():
    def __init__( self, dataset, unit_model ):
        self.dataset = dataset
        self.unit_model = unit_model
        self.lowess_frac = 0.8 # raction of surrounding points in 20mm/s reference estimate
        self.min_n_points = 5
        self.reference_rate = 20 # mm/s


    def analize_units( self, parameter='qc' ):
        for s in self.dataset.soundings:
            # sounding coordinates
            crds = s.get_coordinates() # x, y, z

            # get measured data and registered timestamps
            param_depth, param_data = s.get_data_with_detph( parameter, warped_depth=True, apply_drops=True )
            time_depth, time_data = s.get_data_with_detph( '%', warped_depth=True )

            # link time to depths & calc rates
            param_time = self.calc_depth_time( time_depth, time_data, param_depth )
            param_rate = self.calc_rate( param_time, param_depth )

            # calculate average rate of sounding
            target_rate = self.sounding_target_rate( param_rate ) # tuple: (avg_rate, l_bound and u_bound)
            # outliers are removed before calculating average as they can be very large
            # and clearly wrong an misleading (eg. single point of -1000mm/s - this is just not possible! and irrelevant)

            for u in self.unit_model.units:
                d_top, d_bottom = u.unt_top_bottom_by_coordinate( crds[0], crds[1], return_depth=True )
                u_data, u_depth, u_rate = self.trim_to_unit( param_data, param_depth, param_rate, d_top, d_bottom )

                u_summary = self.unit_summary( u_data, u_depth, u_rate, target_rate )
                if u_summary: # add selected data to summary
                    u_summary['location_name'] = self.dataset.location_name
                    u_summary['position_name'] = s.pos_name
                    u_summary['unit_name'] = u.unit_name
                    u_summary['variable'] = parameter # keep track of what we're looking at
                    u_summary['d_avg'] = u.d_avg[0] # constant: warped depths
                    u_summary['sigma_v0_eff'] = s.soil_stress_model.calc_sigma_v0_eff( u.d_avg[0] )
                    u_summary['sigma_v0'] = s.soil_stress_model.calc_sigma_v0( u.d_avg[0] )
                    u_summary['u0'] = s.soil_stress_model.calc_u0( u.d_avg[0] )
                    u_summary['cv'] = u.cv[0] # constant: warped depths
                    u_summary['rate_desired'] = int(s.target_rate.replace('mm/s',''))
                    u.add_res( u_summary )


    def calculate_reference_values( self, visualize=False ):
        '''
        reference values calculated by smooting value curve from all rates using LOWESS
        and estimaing the value at 20mm/s.  This is done as none of the soundings will
        hit the target rate exactly, and to account for variations between boreholes

        the visualize parameter was created to show why this mehod is useful.
        '''

        self.unit_model.all_vars = [] # list all instances of variables
        for unit in self.unit_model.units:
            for r in unit.res:
                self.unit_model.all_vars.append(unit.res[r]['variable'])
        self.unit_model.all_vars = list( set(self.unit_model.all_vars) )

        # for each variable in each unit, calculate- & register reference and calculate k
        for var in self.unit_model.all_vars:
            for unit in self.unit_model.units:

                # find results with variable
                idx=[]
                for key, some_result in unit.res.items(): # build 
                    if some_result['variable']==var:
                        idx.append( key )

                #perform calculations on those IDs
                rates = np.array( [unit.res[i]['rate_avg'] for i in idx] )
                values = np.array( [unit.res[i]['value_avg'] for i in idx] )

                # sort by rate
                sort_idx = np.argsort( rates )
                rates = rates[ sort_idx ]
                values = values[ sort_idx ]

                reference_estimate = sm.nonparametric.lowess( 
                    values, 
                    rates, 
                    frac=self.lowess_frac, # use 50-80% of points
                    xvals=self.reference_rate #20mm/s
                )[0]
                #reference_estimate = reference_estimate[0]

                for i in idx:
                    unit.res[i]['value_ref'] = reference_estimate
                    if not np.isnan( reference_estimate ):
                        unit.res[i]['k'] = unit.res[i]['value_avg'] / reference_estimate
                        if var=='Bq_':
                            unit.res[i]['k1'] = (unit.res[i]['value_avg']+1) / (reference_estimate+1)
                        if var=='du':
                            unit.res[i]['k_star'] = (unit.res[i]['value_avg']) / (reference_estimate+300)
                    else:
                        unit.res[i]['k'] = np.nan

                if visualize or np.isnan( reference_estimate ): self.visualize_lowess_reference( rates, values, var, unit.unit_name )


    def consolidate_results( self, unit_models ):
        # function to combine results for each unit per position
        # generated to simplify post processing
        cone_diameter = 2*np.sqrt(1000/np.pi) # in mm^2

        all_res = []
        all_units = []

        for some_unit_model in unit_models:
            all_units += some_unit_model.units

        all_vars = self.all_used_variables( all_units )

        for some_unit in all_units:

            positions = self.all_positions_in_unit( some_unit )

            for position_name in positions:
                position_keys = self.get_position_keys( some_unit, position_name )
                has_all_vars, var_keys = self.check_all_vars( all_vars, some_unit, position_keys )
                if has_all_vars:
                    u_p_res = {} # unit_position_results
                    u_p_res['unit'] = some_unit.unit_name
                    u_p_res['site'] = some_unit.res[position_keys[0]]['location_name']
                    u_p_res['position'] = position_name
                    u_p_res['d_avg'] = some_unit.d_avg[0] # constant as depths are warped

                    # following values are averaged but should be close to constant for all cptu_vars
                    u_p_res['sigma_v0'] = self.calc_unit_position_avg( some_unit, position_keys, var='sigma_v0' )
                    u_p_res['u0'] = self.calc_unit_position_avg( some_unit, position_keys, var='u0' )
                    u_p_res['sigma_v0_eff'] = self.calc_unit_position_avg( some_unit, position_keys, var='sigma_v0_eff' )
                    u_p_res['cv'] = self.calc_unit_position_avg( some_unit, position_keys, var='cv' )
                    u_p_res['rate_avg'] = self.calc_unit_position_avg( some_unit, position_keys, var='rate_avg' )
                    u_p_res['rate_std'] = self.calc_unit_position_avg( some_unit, position_keys, var='rate_std' )
                    u_p_res['rate_desired'] = self.calc_unit_position_avg( some_unit, position_keys, var='rate_desired' )
                    u_p_res['V'] = u_p_res['rate_avg'] * cone_diameter / u_p_res['cv']

                    for var, var_key in zip(all_vars, var_keys): # results for each variable
                        u_p_res[var + '_value_avg' ] = some_unit.res[var_key]['value_avg']
                        u_p_res[var + '_value_std' ] = some_unit.res[var_key]['value_std']
                        u_p_res[var + '_value_ref' ] = some_unit.res[var_key]['value_ref']
                        u_p_res[var + '_k' ] = some_unit.res[var_key]['k']
                        if 'k1' in some_unit.res[var_key]: # try to fix presentation of PP against rate
                            u_p_res[var + '_k1' ] = some_unit.res[var_key]['k1']
                        if 'k_star' in some_unit.res[var_key]:
                            u_p_res[var + '_k_star' ] = some_unit.res[var_key]['k_star']

                    #inspired by
                    # Lehane et al. 2009
                    u_p_res['qt_sigma_v0_eff'] = u_p_res['qt_value_avg'] / u_p_res['sigma_v0_eff']
                    u_p_res['fs_sigma_v0_eff'] = u_p_res['fs_value_avg'] / u_p_res['sigma_v0_eff']
                    u_p_res['qt_sigma_v0'] = u_p_res['qt_value_avg'] / u_p_res['sigma_v0']

                    # schneider et al. 2008
                    u_p_res['du_sigma_v0_eff'] = u_p_res['du_value_avg'] / u_p_res['sigma_v0_eff']

                    all_res.append( u_p_res )
        return all_res


    def all_positions_in_unit( self, unit ): 
        # returns list of all positions that have results in unit
        positions = []
        for some_res_value in unit.res.values():
            positions.append(some_res_value['position_name'])
        positions = list( set( positions) )
        positions.sort()

        return positions


    def all_used_variables( self, units ): 
        # returns list with all variables used in any of the units
        list_of_vars = []
        for unit in units:
            for some_res in unit.res.values():
                list_of_vars.append(some_res['variable'])

        unique_var_list = [] # retains initial order of var apperances
        [ unique_var_list.append(item) for item in list_of_vars if item not in unique_var_list ]

        return unique_var_list


    def get_position_keys( self, some_unit, position_name ): 
        # returns list of keys for results for a single position
        keys = []
        for res_ind, res_value in some_unit.res.items():
            if res_value['position_name']==position_name:
                keys.append( res_ind )
        keys.sort()
        return keys


    #has_all_vars, var_idx = self.check_all_vars
    def check_all_vars( self, all_vars, some_unit, position_keys ): 
        # returns True if all vars are present for position in unit, else False
        # along with indexes of (only) found vars
        ans = [ False ] * len( all_vars )
        ans_keys = []
        for i, var in enumerate( all_vars ):
            for key in position_keys:
                if some_unit.res[key]['variable']==var:
                    ans[i]=True
                    ans_keys.append( key )
                    break
        return all(ans), ans_keys


    def calc_unit_position_avg( self, some_unit, position_keys, var ):
        # average value of position variable through unit from avg of all variable estimates
        # varies a little as sensors have different sampling frequencies.
        # eg. rate estimates will vary a little between sensor registrations for qc, fs or u2.

        var_values = []
        for p_key in position_keys:
            var_values.append( some_unit.res[p_key][var])
        return np.average(np.array(var_values))


    def transform_to_csv( self, results ):
        keys = list( results[0].keys() )
        csv = ','.join(keys)

        for som_result in results:
            csv += '\n'
            for key in keys:
                csv += str( som_result[key] ) + ','
            csv = csv[0:-1]
        return csv


    def write_to_csv( self, f_name, results ):
        csv = self.transform_to_csv( results )

        with open( f_name, 'wb') as file:
            file.write( csv.encode("utf-8") )


    def calc_rate( self, times, depths ):
        time_incr = np.diff( times ) # in seconds
        depth_incr = np.diff( depths ) * 1000 # to millimeters

        # remove nans by index from both x&y
        #nan_indexes = np.isnan( time_incr ) | np.isnan( depth_incr )
        #time_incr, depth_incr = time_incr[~nan_indexes], depth_incr[~nan_indexes]

        rate = depth_incr / time_incr # nans cause RuntimeWarning - ignored
        rate = np.append( [rate[0]], rate ) # duplicate first value

        return rate


    def calc_depth_time( self, depths, times, param_depths ):
        param_times = np.interp( param_depths, depths, times ) # millisecs

        return param_times/1000 # sec


    def trim_to_unit( self, data, depth, rate, d_top, d_bottom ):
        # drop ranges with nans (from warp:drop_ranges)
        _, nan_mask = self.drop_nans( data )
        data, depth, rate = data[~nan_mask], depth[~nan_mask], rate[~nan_mask]

        from_idx = np.searchsorted( depth, d_top )
        to_idx = np.searchsorted( depth, d_bottom )

        # construct trimmed array ( start + idx range from first (with!) to last (without!) + end )
        trim_depth = np.concatenate( ([d_top], depth[from_idx:to_idx], [d_bottom]) )

        trim_data = np.interp( trim_depth, depth, data )
        trim_rate = np.interp( trim_depth, depth, rate )

        return trim_data, trim_depth, trim_rate


    def sounding_target_rate( self, rate):
        # remove nans
        rate, _ = self.drop_nans( rate )

        x = np.arange( len(rate) )

        # remove outliers
        f_rate, f_lbound, f_ubound = self.iqr_outliers( rate )

        indices = np.where(np.in1d(rate, f_rate))[0]
        x1=x[indices]

        avg_rate = np.average( f_rate )

        if False:
            fig, ax = plt.subplots()
            ax.plot(x,rate)
            ax.plot([x[0], x[-1]],[avg_rate, avg_rate])
            ax.plot([x[0], x[-1]],[f_lbound, f_lbound])
            ax.plot([x[0], x[-1]],[f_ubound, f_ubound])
            plt.show()

        return avg_rate, f_lbound, f_ubound


    def iqr_outliers( self, vals, q_range=[25,75] ):
        q1, q3 = np.percentile( vals, q_range )
        iqr = q3 - q1

        lbound = q1 - 1.5*iqr
        ubound = q3 + 1.5*iqr

        mask = (vals > lbound) & (vals < ubound)

        return vals[mask], lbound, ubound


    def drop_nans( self, vals ):
        nan_mask = np.isnan( vals )
        return vals[~nan_mask], nan_mask


    def unit_summary( self, data, depth, rate, target_rate ):
        # filter out rates outside outlier detection mask
        mask = ( rate>target_rate[1] ) & ( rate<target_rate[2] ) # target_rate:=(avg_rate, l_bound and u_bound)
        f_data, f_depth, f_rate = data[mask], depth[mask], rate[mask]

        res = {}
        if len( f_data ) > self.min_n_points:
            #res[ 'rate' ]      = np.array2string( f_rate, precision=8, separator=',', suppress_small=True ) # will be saved to JSON
            #res[ 'value' ]     = np.array2string( f_data, precision=4, separator=',', suppress_small=True )

            res[ 'rate_avg' ]  = np.average( f_rate )
            res[ 'rate_std' ]  = np.std( f_rate )
            res[ 'value_avg' ] = np.average( f_data )
            res[ 'value_std' ] = np.std( f_data )


            if False: # check if everything is working as it should
                n = len(f_rate)
                x = np.arange( n )
                f_r_avg = np.array( [res[ 'rate_avg' ]] * n )
                f_d_avg = np.array( [res[ 'value_avg' ]] * n )

                fig, ax1 = plt.subplots()
                ax1.scatter( x, f_rate, c='r', label='rate' ) 
                ax1.plot( x, f_r_avg, ls='-', marker=None, c='r', label='rate' )
                ax1.plot( x, f_r_avg-res[ 'rate_std' ], ls='--', marker=None, c='r', label='rate' )
                ax1.plot( x, f_r_avg+res[ 'rate_std' ], ls='--', marker=None, c='r', label='rate' )

                ax2 = ax1.twinx()
                ax2.scatter( x, f_data, c='b', label='data' )
                ax2.plot( x, f_d_avg, c='b', label='data' )
                ax2.plot( x, f_d_avg-res[ 'value_std' ], ls='--', c='b', label='data' )
                ax2.plot( x, f_d_avg+res[ 'value_std' ], ls='--', c='b', label='data' )

                plt.legend()
                plt.show()
        return res

    def visualize_lowess_reference( self, rate, value, variable, unit ):
    # used to document how the reference value for v=20 was selected.
        if variable!='qt': return #'Qt_'
        fontsize_titles=20
        fontsize_ticks=18
        site_nr = self.dataset.location
        size=70
        def marker_style( label ):
            markers = {
                0: ( 'o', 1 ),
                1: ( 'v', 1 ),
                2: ( 'X', 1.3 )
            }
            if label in markers: return markers[label]
            return 'o', 1

        def marker_color( label ):
            edgecolor = ( .1, .1, .1 )
            colors={
                'model': (237/255,28/255,46/255), # NPRA red
                0: (93/255,184/255,46/255), # NPRA green
                1: (0/255,142/255,194/255), # NPRA blue
                2: (255/255,150/255,0/255), # NPRA orange
            }        
            if label in colors: return ( colors[label], edgecolor )

        m, msf = marker_style( site_nr )
        mc, mc_edge = marker_color( site_nr )

        frac=self.lowess_frac
        it=3
        reference_estimate = sm.nonparametric.lowess(value, rate, frac=frac, xvals=self.reference_rate, it=it)
        regression = sm.nonparametric.lowess(value, rate, frac=frac, xvals=rate, it=it)

        fig, ax = plt.subplots()
        ax.scatter( rate, value, s=size*msf, marker=m, c=mc, edgecolors=mc_edge, label='averages in unit ' + str(unit), zorder=1 )
        ax.scatter( self.reference_rate, reference_estimate, s=size*msf, c='r', edgecolors=mc_edge, label='calculated reference', zorder=3 )
        ax.plot( rate, regression, c='r', lw=2,label='lowess, f=.8', zorder=2 )
        ax.set_xlabel( 'Penetration rate, v mm/s', fontsize=fontsize_titles )
        ax.set_ylabel( 'q' + r'$_t$' + ' (kPa)', fontsize=fontsize_titles )
        ax.tick_params( axis='x', labelsize=fontsize_ticks )
        ax.tick_params( axis='y', labelsize=fontsize_ticks )
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        plt.legend( fontsize=fontsize_ticks*1 )
        plt.tight_layout()
        plt.show()
        a=1