'''
    Class for parsing cpt log-files
    used by the cpt class
'''

import numpy as np
import scipy
from scipy import stats
from smoothers import lowess
import matplotlib.pyplot as plt


class log_file():
    def __init__( self, reference, file_contents ):
        self.lowess_interval = 0.3
        self.data = {}
        self.zero_1 = {}
        self.zero_2 = {}

        self.cpt_ref = reference
        self.raw_header, self.raw_data = file_contents.split( '<Received Data>\n' )
        self.parse_logfile() # read and organize data
        self.update() # update log_data with data from cpt file


    def update( self ):
        self.shift_scale_depths( )

        if False: # these removed lines were used to find meaning of registrations
            self.find_fit_to_cpt_data( self.cpt_ref ) # analysis attempt (partial success only)
            self.to_file() # data imported to Excel this way for further analyses

        self.consolidate_registrations() # combine registrations stored in different streams
        self.subtract_zeroes_and_scale()
        self.calculate_stops()
        self.remove_stops()
        #self.fill_stops() # does nothing
        self.calc_rate()
        self.calc_log_freq()
        self.geometrical_depth_correction()
        self.calc_cpt_params()
        self.calc_lowess_params()


    def calc_cpt_params( self ):
        # cone correction factors
        alfa = self.cpt_ref.calibration['alfa'] if 'alfa' in self.cpt_ref.calibration else -1
        beta = self.cpt_ref.calibration['beta'] if 'beta' in self.cpt_ref.calibration else -1


        # u2 related variables
        u0_u = self.cpt_ref.soil_stress_model.calc_u0(self.cpt_data['u']['d'] )
        sig_v0_eff_u = self.cpt_ref.soil_stress_model.calc_sigma_v0_eff(self.cpt_data['u']['d'] )

        self.cpt_data['du'] = { 
            'd':self.cpt_data['u']['d'], 
            'value': self.cpt_data['u']['value'] - u0_u } # u2-u0

        self.cpt_data['du_sig_v0_eff'] = { 
            'd':self.cpt_data['u']['d'], 
            'value': self.cpt_data['du']['value'] / sig_v0_eff_u } # du/sigma_v0'


        # qc related variables
        u2_q = np.interp( self.cpt_data['qc']['d'], self.cpt_data['u']['d'], self.cpt_data['u']['value'])
        sig_v0_q = self.cpt_ref.soil_stress_model.calc_sigma_v0(self.cpt_data['qc']['d'] )
        sig_v0_eff_q = self.cpt_ref.soil_stress_model.calc_sigma_v0_eff(self.cpt_data['qc']['d'] )

        self.cpt_data['qt'] = { 
            'd':self.cpt_data['qc']['d'], 
            'value': self.cpt_data['qc']['value'] + ( 1-alfa ) * u2_q }

        self.cpt_data['qn'] = { 
            'd':self.cpt_data['qc']['d'], 
            'value': self.cpt_data['qt']['value'] - sig_v0_q }
        qn_u = np.interp( self.cpt_data['u']['d'], self.cpt_data['qn']['d'], self.cpt_data['qn']['value']) # for Bq

        self.cpt_data['qe'] = { 
            'd':self.cpt_data['qc']['d'], 
            'value': self.cpt_data['qt']['value'] - u2_q }

        self.cpt_data['Qt_'] = { 
            'd':self.cpt_data['qn']['d'], 
            'value': self.cpt_data['qn']['value'] / sig_v0_eff_q }

        self.cpt_data['Bq_'] = { 
            'd':self.cpt_data['du']['d'], 
            'value': self.cpt_data['du']['value'] / qn_u }


        # fs related variables
        u2_f = np.interp( self.cpt_data['fs']['d'], self.cpt_data['u']['d'], self.cpt_data['u']['value'])
        du_f = np.interp( self.cpt_data['fs']['d'], self.cpt_data['du']['d'], self.cpt_data['du']['value'])
        qt_F = np.interp( self.cpt_data['fs']['d'], self.cpt_data['qt']['d'], self.cpt_data['qt']['value'])
        qn_F = np.interp( self.cpt_data['fs']['d'], self.cpt_data['qn']['d'], self.cpt_data['qn']['value'])

        self.cpt_data['ft'] = { # formula from sgi-i15 - an approximation as u3 is not measured!
            'd':self.cpt_data['fs']['d'],
            'value': self.cpt_data['fs']['value'] - ( beta*u2_f + 0.3*du_f*((1-alfa)/15-beta) ) }

        self.cpt_data['Fr_'] = { 
            'd':self.cpt_data['fs']['d'], 
            'value': self.cpt_data['fs']['value'] / qn_F * 100 }
        
        self.cpt_data['Rf_'] = { 
            'd':self.cpt_data['fs']['d'], 
            'value': self.cpt_data['fs']['value'] / qt_F * 100 }


    def calc_lowess_params( self ):
        smoother = lowess( delta=self.lowess_interval )
        res = {}
        for param in self.cpt_data:
            if param=='o': # no smoothing for temperature
                res['lowess_' + param ] = { 'd': self.cpt_data[param]['d'],'value': self.cpt_data[param]['value'] }
                res['res_' + param ] = { 'd': self.cpt_data[param]['d'],'value': self.cpt_data[param]['value'] * 0 }
                continue

            smoother.fit( self.cpt_data[param]['d'], self.cpt_data[param]['value'] )
            res[ 'lowess_' + param ] = {
                'd': self.cpt_data[param]['d'],
                'value': smoother.predict()
            }
            res[ 'res_' + param ] = {
                'd': self.cpt_data[param]['d'],
                'value': self.cpt_data[param]['value'] - res[ 'lowess_' + param ]['value']
            }

        self.cpt_data.update( res )


    def calculate_stops( self ):
        if 'Z' not in self.data: return
        self.breaks = self.data['Z'].get_breaks() # find sounding stops


    def remove_stops( self ):
        # construct & apply masks
        for channel in self.cpt_data: 
            d = self.cpt_data[channel]['d']
            for some_break in self.breaks:
                mask = self.get_mask( some_break, d )
                self.cpt_data[channel]['value'][mask] = np.nan # add nans where stops interfere

            nan_mask = ~np.isnan(self.cpt_data[channel]['value'])
            self.cpt_data[channel]['value'] = self.cpt_data[channel]['value'][nan_mask] # drop nans
            self.cpt_data[channel]['d'] = self.cpt_data[channel]['d'][nan_mask]
            self.cpt_data[channel]['time'] = self.cpt_data[channel]['time'][nan_mask]


    def fill_stops( self ):
        pass


    def geometrical_depth_correction( self ):
        offset_id = 'offset_'

        all_calibration_keys = list( self.cpt_ref.calibration.keys() )
        offset_keys = [ k for k in all_calibration_keys if offset_id in k ]
        for offset_key in offset_keys:
            param = offset_key.split('_')[1]
            self.cpt_data[param]['d'] -= self.cpt_ref.calibration[offset_key] #apply shift


    def get_mask( self, some_break, d ):
        h = 0.07 # remove 10cm of data
        delta = h/2
        shifted_break = some_break + delta * 0.9
        return np.abs(d-shifted_break)<delta


    def calc_rate( self ):
        param = 'qc'

        d = self.cpt_data[param]['d']
        t = self.cpt_data[param]['time'] / 1000 # secs
 
        sorted_ind = np.argsort( t ) # sort
        d = d[sorted_ind]
        t = t[sorted_ind]

        t_inc = np.diff( t ) # calculate increments
        d_inc = np.diff( d )

        cont_indices = t_inc<5  # no breaks
        breaks = ~cont_indices

        t_inc_cont = t_inc[ cont_indices ]
        t_inc_breaks = t_inc[breaks]
        t_brekas = sum(t_inc_breaks)

        rates = d_inc / t_inc * 1000 # m/s to mm/s
        ret_rates = np.insert(rates, 0, rates[0]) # duplicate 1st value at start of list
        std_rates = rates[rates>2] # slowest rate in project 5mm/s

        tot_penetration_depth = d[-1]-d[0]
        T_cont = np.sum( t_inc_cont ) # total continuous penetration time
        T_total_cont = t[-1]-t[0]-t_brekas

        avg_rate = tot_penetration_depth / T_cont * 1000 # unsure why this is too high!
        avg_rate_w_breaks = np.average(rates)

        std_rates = rates[rates>2] # slowest rate in project 5mm/s
        st_dev = np.std( std_rates )
        avg_std_rates = np.average( std_rates )

        self.rate = {
            'avg_rate': avg_std_rates, #avg_rate,
            'rate_st_dev': st_dev,
            'avg_rate_w_breaks': avg_rate_w_breaks,
            'rates': ret_rates
        }


    def calc_log_freq( self ): # freq for whole sounding. breaks >5s removed
        self.avg_log_freq = {}

        for param in self.cpt_data:
            t = self.cpt_data[param]['time'] / 1000 # secs
            t.sort()

            t_inc = np.diff( t ) # increments

            t_inc_cont = t_inc[ t_inc < 5 ]
            T_cont = np.sum( t_inc_cont ) # penetration time omitting breaks

            no_breaks = len(t_inc) - len( t_inc_cont ) # number of stops >= 5s

            if param in ['qc','fs','u']:
                samples = len(t)-no_breaks
                avg_freq = samples / T_cont
                self.avg_log_freq[ param ] = avg_freq


    def to_file( self ):
        output = ''
        for stream in self.data:
            #a = self.data[stream].data
            for i in range(len(self.data[stream].data)):
                tmp = ' '.join(self.data[stream].data[i].astype(str).tolist() )
                output += stream + str(i) + ':' + tmp + '\n'

        output += '\n\n'
        for var in self.cpt_ref.data:
            tmp = ' '.join(self.cpt_ref.data[var].astype(str).tolist() )
            output += var + ' ' + tmp + '\n'


        with open('output.txt', 'w') as file:
            file.write(output)


    def subtract_zeroes_and_scale( self ):        
        keys_to_zero = ['qc', 'fs', 'u']

        reg_bits = 17
        offset = 385
        base_var = ( 2**reg_bits + offset ) / 10**6

        base_scale = 1#1.718e-4**0.5 # base scale estimate
        scale_factor = { 'qc':base_var/100,'fs':base_var/5,'u':base_var/10 } # previous_estimate 'qc': 0.0013152304, 'fs': 0.026304608, 'u': 0.013152304

        stream_keys = self.get_stream_keys()

        for key, value in stream_keys.items():
            if key in keys_to_zero:
                all_items = list(value.items())
                first_channel = str(all_items[0][0]) + str(all_items[0][1])
                zero1 = self.zero_1['zero-readings'][first_channel]
                zero2 = self.zero_2['zero-readings'][first_channel]
                self.cpt_data[key]['value'] -= zero1

                scale = 1 / (self.cpt_ref.calibration['scale_fact_' + str(key)] * (scale_factor[key]*base_scale ))
                self.cpt_data[key]['value'] *= scale
                self.cpt_data[key]['zero'] = (zero2-zero1) * scale


    def consolidate_registrations( self ):
        stream_keys = self.get_stream_keys()

        self.cpt_data = {} # combine data from multiple streams
        for k in stream_keys:
            combined = False
            for stream in stream_keys[k]:
                if stream in self.data: # different streams between rigs ( A, B, C, F, Z,...)
                    if k in self.cpt_data: # append more data to same cptu variable (new depth-value-time registrations)
                        combined = True
                        self.cpt_data[k]['d'] = np.append( self.cpt_data[k]['d'], self.data[stream].data[2])
                        self.cpt_data[k]['value'] = np.append( self.cpt_data[k]['value'], self.data[stream].data[stream_keys[k][stream]])
                        self.cpt_data[k]['time'] = np.append( self.cpt_data[k]['time'], self.data[stream].data[-1])
                    else: # new cptu variable
                        self.cpt_data[k] = {
                            'd': self.data[stream].data[2],
                            'value': self.data[stream].data[stream_keys[k][stream]],
                            'time': self.data[stream].data[-1],
                        }

            if combined: # sort combined cptu variables by ascending depth
                sorting_key = np.argsort( self.cpt_data[k]['d'] )
                self.cpt_data[k]['d'] = self.cpt_data[k]['d'][sorting_key]
                self.cpt_data[k]['value'] = self.cpt_data[k]['value'][sorting_key]
                self.cpt_data[k]['time'] = self.cpt_data[k]['time'][sorting_key]

            # remove multiple registration from same depth (keep first)
            depths_to_keep = self.cpt_data[k]['d'][:-1] != self.cpt_data[k]['d'][1:] # neighbor is unique eg. [False, False, ... ,True]
            depths_to_keep = np.insert( depths_to_keep, 0, True ) # add first item as unique [True, False, False, ... ,True]
            depths_to_keep = np.where( depths_to_keep )[0] # calculate indexes of Trues [0, 34, 35, 38, ..., 1248]

            self.cpt_data[k]['d'] = self.cpt_data[k]['d'][depths_to_keep]
            self.cpt_data[k]['value'] = self.cpt_data[k]['value'][depths_to_keep]
            self.cpt_data[k]['time'] = self.cpt_data[k]['time'][depths_to_keep]


    def get_stream_keys( self ):
        # returns a key to link stream values (A,stream_1,stream_2,...,stream_n) to cptu parameters
        if hasattr( self.data['C'], 'key'): # build key dict from data_analysis (highest r-match results)
            stream_keys = {}
            for stream in self.data:
                if stream == 'Z': continue # discard depth registrations
                some_key = self.data[stream].key # ex. {0: ['qc', some_r_value], 1:... }
                for k in some_key: # 0, 1, 3, 4
                    if some_key[k][0] in stream_keys: # 'qc', 'fs', ...
                        stream_keys[some_key[k][0]][stream] = k # update existing dict
                    else:
                        stream_keys[some_key[k][0]] = {stream: k} # new dict

        else: # combined dict from experiences
            stream_keys = {
                'qc':{
                    'C': 0,
                },
                'fs':{
                    'F': 1
                },
                'u':{
                    'F': 0,
                    'A': 1,
                },
                'ta':{
                    'C': 1,
                },
                'b':{
                    'C': 3,
                    'F': 3,
                    'A': 3,
                    'B': 3,
                },
                'o':{
                    'A': 0,
                },
                '%':{
                    'C': -1, # 4 or 6
                    'F': -1,
                    'A': -1,
                    'B': -1,
                },
            }
        return stream_keys


    def find_fit_to_cpt_data( self, cpt_reference ):
        for stream in self.data:
            if stream == 'Z':
                self.breaks = self.data[stream].get_breaks()
            else:
                self.data[stream].find_best_fit_to_cpt_data( cpt_reference )


    def remove_empty_sets( self ):
        number_of_records = []
        for stream in self.data:
            number_of_records.append( len(self.data[ stream ].data[0]) )

        n_cutoff = int( max(number_of_records)*0.01 ) # empty if <1% of largest set
        self.data = { stream:data for stream,data in self.data.items() if len(self.data[stream].data[0])>n_cutoff }


    def shift_scale_depths( self ):
        self.starting_depth = self.cpt_ref.data['d'][0]
        for stream in self.data:
            self.data[stream].shift_scale_depth( self.starting_depth )


    def parse_logfile( self ):
        self.raw_header = self.sanitize_string( self.raw_header )
        self.calc_zero( self.zero_1, self.raw_data.split( 'Command: DRON' )[0] ) # pass string with zero reading
        self.calc_zero( self.zero_2, self.raw_data.split( '<END BUTTON' )[1] )
        self.get_data()  # isolate data block
        self.cast_to_floats() # for all streams:  str->float
        self.remove_empty_sets()


    def cast_to_floats( self ):
        for stream in self.data:
            self.data[stream].cast_to_floats()


    def sanitize_string( self, some_string ):
        some_string = some_string.replace( '[^C]', '' )
        some_string = some_string.replace( '[^G]', '' )
        some_string = some_string.replace( '[CR]', '' )
        for _ in range(5): some_string = some_string.replace( '\n\n', '\n' )

        return some_string


    def get_zero_before( self ):
        # trim to data
        zero_1 = self.raw_data.split( 'Command: DRON' )[0]
        zero_1 = self.isolate_lines_with_data( zero_1 )
        zero_1 = self.sanitize_string( zero_1 )

        zero1_lines = zero_1.split('\n')
        for line in zero1_lines:
            self.add_line_of_data( self.zero_1, line )

        self.zero_1['zero-readings'] = {}

        for stream in self.zero_1:
            if stream=='Z' or stream=='zero-readings': continue
            for i in range( len(self.zero_1[stream].data) ):
                vals = self.zero_1[stream].data[i]
                val =  self.calc_zero_value( vals )
                self.zero_1['zero-readings'][ stream + str(i) ] = val


    def calc_zero_value( self, readings ):
        readings = np.sort(readings).astype(np.float64)
        if len(readings)>3:
            readings = readings[1:-1]

        return np.average( readings )


    def get_zero_after( self ):
        zero_2 = self.raw_data.split( '<END BUTTON' )[1]
        zero_2 = self.isolate_lines_with_data( zero_2 )
        zero_2 = self.sanitize_string( zero_2 )

        zero2_lines = zero_2.split('\n')
        for line in zero2_lines:
            self.add_line_of_data( self.zero_2, line )
        

    def calc_zero( self, ref, string_w_zero):
        string_w_zero = self.isolate_lines_with_data( string_w_zero )
        string_w_zero = self.sanitize_string( string_w_zero )

        zero_lines = string_w_zero.split('\n')
        for line in zero_lines:
            self.add_line_of_data( ref, line )

        ref['zero-readings'] = {}

        for stream in ref:
            if stream=='Z' or stream=='zero-readings': continue
            for i in range( len(ref[stream].data) ):
                vals = ref[stream].data[i]
                val =  self.calc_zero_value( vals )

                ref['zero-readings'][ stream + str(i) ] = val


    def get_data( self ):
        # trim to data
        data = self.raw_data.split( 'Command: DRON' )[1]
        data = data.split( '<END BUTTON' )[0]

        data = self.isolate_lines_with_data( data )
        data = self.sanitize_string( data )
        data_lines = data.split('\n')
        for line in data_lines:
            self.add_line_of_data( self.data, line )


    def isolate_lines_with_data( self, multiline_string ):
        # removes lines from string if they do not contain n splitter chars
        splitter = ','
        n = 3

        ret_string = ''

        lines = multiline_string.split( '\n' )
        for line in lines:
            if ( len(line)-len(line.replace(splitter,'')) ) > n:
                ret_string += line + '\n' # only lines with >n splitters

        return ret_string[:-1] # removes last newline


    def add_line_of_data( self, target, line ):
        items = line.split(',')
        if items[0] in target:
            target[items[0]].add_line( items )
        else:
            target[items[0]] = log_file.data_stream( items )




    class data_stream():
        def __init__( self, items ):
            self.ID = items[0] # extract ID
            self.num_data_items = len(items)-1

            self.init_data( items ) # create np.arrays
            self.add_line( items ) # add first line


        def init_data( self, items ):
            self.data = []
            for i in range( 1, len(items) ):
                self.data.append(np.array([]))


        def add_line ( self, items ):
            data_items = items[1:]
            n = len(data_items) - self.num_data_items

            if n>0: # more columns than in previous line(s)
                for i in range( n ):
                    self.init_data( items ) # start over from here!
                    self.num_data_items = len(items)-1

            elif n==0:
                for i in range( len(data_items) ):
                    self.data[i] = np.append( self.data[i], data_items[i] )
            # using these if-s skips lines with missing data: n<0:


        def cast_to_floats( self ): # casts values in complete streams:  str->float
            for i in range( len(self.data) ):
                n_max = len(max(self.data[i], key=len)) # date checker
                if n_max > 9: # save and shift by start_time ( is a ~10 or 17 digit number )
                    self.data[i] = self.data[i].astype(np.int64)
                    self.start_time = self.data[i][0] # preserves decimals
                    self.data[i] -= self.start_time

                self.data[i] = self.data[i].astype(np.float64) # finally cast


        def shift_scale_depth( self, depth_shift ):
            # scales depth from mm to m.  depth_shift=predrilling ( predrilling depth not stored in log file )
            self.data[2] = self.data[2]/1000 + depth_shift


        def get_breaks( self ):
            z_reg = self.data[2]
            unique_depths, counts = np.unique( z_reg, return_counts=True )

            # calculate threshold (remove 1's and 2's)
            # depths_filtered = unique_depths[ counts>2 ] 
            counts_filtered = counts[ counts>2 ] 
            count_threshold = np.percentile( counts_filtered, 25)/2 # half 25th percentile

            return unique_depths[ counts>count_threshold ]


        def closest_depths( self, x1, x2, n=0.05 ): # returns ids for % closest points (sorted by shorter)
            short_first = True # find shortest array
            if len( x1 ) < len( x2 ):
                short = x1
                long = x2
            else:
                short_first = False
                short = x2
                long=x1

            n_points = int( len(short) * n ) # % of points from shorter curve

            i_short=np.arange( 0, len(short), dtype=np.int ) # indices of shorter array

            i_long = np.empty( [], dtype=np.int )
            i_long = np.delete( i_long, np.s_[:] )
            for ps in short:
                idx = ( np.abs(long - ps) ).argmin() # indice for closest match to ps
                i_long = np.append( i_long, idx )

            dists = abs( short-long[i_long] ) # distance between each closest match
            i_sort = dists.argsort() # indices of sorted distances (smallest to largest)

            i_sort = i_sort[:n_points]
            i_sort.sort() # sort the soring array :)

            if short_first: # return ids for desired % of points
                return i_short[i_sort], i_long[i_sort]
            return i_long[i_sort], i_short[i_sort]


        def find_best_fit_to_cpt_data( self, cpt_reference ):
            # check for matching depths
            d_stream = self.data[2]
            d_cpt = cpt_reference.data['d']
            self.i_stream, self.i_cpt = self.closest_depths( d_stream, d_cpt, n=0.05 )

            solution = {}

            for i in range(len(self.data)):
                if i==2: continue # ignore dept
                sel_stream_data = self.data[i][ self.i_stream ] # filtered by depth-match
                for data_header in cpt_reference.data:
                    #if data_header == 'd': continue # ignore depth
                    if data_header not in ['qc','u','fs','b','ta','o', '%']: continue
                    sel_cpt_data = cpt_reference.data[data_header][self.i_cpt] # also filtered by depth-match

                    r, _ = scipy.stats.pearsonr( sel_stream_data, sel_cpt_data )
                    if r>0.5:
                        if i in solution:
                            solution[i][0].append( data_header )
                            solution[i][1].append( r )
                        else:
                            solution[i] = [ [data_header], [r] ]

                        #print( str(self.ID) + ': ' + str( i ) + ' - ' + str(data_header) + '  ###  Good fit: r=' + str(round(r,5)) )

            for key in solution:
                i = np.array( solution[key][1] ).argmax()
                solution[key] = [solution[key][0][i], solution[key][1][i]]
                print( str(self.ID) + str(key) + ': ' + str(solution[key][0]) + ' ' + str(solution[key][1]) )

            self.key = solution


        def check_fit( self, x1, y1, x2, y2 ):
            y_n1, y_n2 = self.interpolate_to_match_length( x1, y1, x2, y2 ) # returns equal lengh arrays
            r, _ = scipy.stats.pearsonr(y_n1, y_n2)
            return r


        def interpolate_to_match_length( self, x1, y1, x2, y2 ):
            # get x-definition
            min_x = min( x1.min(), x2.min() )
            max_x = max( x1.max(), x2.max() )
            n_max = max( len(x1), len(x2) )

            # define x
            x_ = np.linspace( min_x, max_x, n_max )

            # calculate y
            y_1 = np.interp( x_, x1, y1 )
            y_2 = np.interp( x_, x2, y2 )

            return y_1, y_2


        def robust_scaler( self, x ):
            p25 = np.percentile( x, 25 )
            p50 = np.percentile( x, 50 ) # median
            p75 = np.percentile( x, 75 )
            return ( x - p50 )/ ( p75-p25 )


        def scale_to_fit( self, x1, x_ref ):
            x = x1 - np.median(x1)


        def plot( self, d=None ):

            d_set = None
            n_charts = len( self.data )

            fig, axs = plt.subplots( 1, n_charts, sharey=True )
            axs[0].invert_yaxis()
            fig.suptitle( 'Data stream: ' + str(self.ID) )

            if d and self.ID in d:
                d_set = d[ self.ID ]
                cpt_ref = d[ 'ref' ]

            for i in range( len(self.data) ):
                axs[i].plot( self.data[i], self.data[2] )
                axs[i].set_title( str(i) )

                if d_set and i in d_set:
                    scaled_x = self.scale_to_fit( cpt_ref.data[d_set[i]], self.data[i] )
                    axs[i].plot( scaled_x, cpt_ref.data['d'] )
            plt.show()