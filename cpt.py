import os
import numpy as np
from smoothers import lowess
from alfa_beta import calibration_data
from cpt_log import log_file

# class to read CPT files and parse contents.
# expanded to also read cpt log files via external class cpt_log.


class cpt( ):
    def __init__( self, filename, soil_stress_model, read_log_file=True, ignore_calibration=False ):
        self.lowess_interval = 0.3

        self.calibration = {}
        self.depth_warp_def = { 'd_from': np.array([0]), 'd_to': np.array([0]) } # init with no warp
        self.remove_sounding = False

        self.soil_stress_model = soil_stress_model
        self.cpt_calibrations = calibration_data() # import all calibrations - used in update()

        self.filename = filename
        self.log_file_path = self.check_for_log_file()

        self.pos_name = self.get_posname( filename )

        raw_file = self.replace_unwanted_chars( self.read_file( self.filename ) ) # replace null char

        self.raw_header, self.raw_data = self.split_raw_file( raw_file ) # only first sounding in file
        if self.remove_sounding: return # early validation/exit

        self.parse_headers()
        self.parse_data()

        self.remove_sounding = self.sounding_has_problems()

        if self.remove_sounding: return # exit after validation

        if ignore_calibration:  # no interpretation should be done
            self.calibration = self.cpt_calibrations.standard_certificate( self.headers['hn'], self.headers['hd'] )
        else: self.calibration = self.cpt_calibrations.get_sertificate( self.headers['hn'], self.headers['hd'] )


        self.geometrical_depth_correction()
        self.update()

        if self.log_file_path and read_log_file: # better data
            self.log_file = log_file( self, self.read_file( self.log_file_path ) ) # reference self & parse log


    def sounding_has_problems( self ): # returns True if problems are found
        if not hasattr( self, 'data' ): return True # need data
        if not hasattr( self, 'headers' ): return True # need headers
        if len(self.data)==0: return True
        if len(self.headers)==0: return True

        if any(item not in self.data for item in ['d', 'qc', 'fs', 'u']): return True
        if len(self.data['d']) < 5: return True # registration at at least 5 registrations

    def calibrate_depth_warper( self ): # used with dtw script
        '''
        Sets depth warping arrays (from->to) to be used for interpolation
        warping omitted if warp path is not present
        '''

        d= self.data['d']
        if hasattr( self, 'log_file'): d=self.log_file.cpt_data['qc']['d'] # overwrite

        x = np.copy( self.get_warping_depth() )
        y = np.copy( x )

        if hasattr( self, 'warp_path') and hasattr( self, 'warping_d'):
            for item in self.warp_path:  #(idx_to_change, idx_of_change)
                y[item[0]] = self.warping_d[item[1]]

        if hasattr( self, 'depth_shifts') and self.depth_shifts: # depth shifts contain items
            x = np.array([])
            y = np.array([])
            for item in self.depth_shifts:
                x = np.append( x, item[0] )
                y = np.append( y, item[1] )

        self.depth_warp_def = { 'd_from': x, 'd_to': y }


    def set_warp( self, warp, drops ):
        d_from = np.array([w[0] for w in warp])
        d_to = np.array([w[1] for w in warp])
        self.depth_warp_def = { 'd_from': d_from, 'd_to': d_to }
        self.drops = drops


    def warp_depths( self, d_from, inverse_warp=False ):
        ''' warps depths based on definition in self.depth_warp_def
        
            warps modelled using "delta=d_to-d_from" because of definition of np.interp function, where values below the definition range take the first value
            and values above the definition take the last. modelling the delta uses the last warp definition, and will vary with varying registrations.
            
            the direct approach (eg. np.interp(d_from,self.depth_warp_def['d_from'],self.depth_warp_def['d_to']) doesn't give as good a result outside defined range)
        '''
        x_fr = self.depth_warp_def['d_from']
        delta = self.depth_warp_def['d_to'] - self.depth_warp_def['d_from']

        if inverse_warp:
            x_fr = self.depth_warp_def['d_to']
            delta = self.depth_warp_def['d_from'] - self.depth_warp_def['d_to']
        
        return self.delta_warper( d_from, x_fr, delta )


    def delta_warper( self, x, x_from, delta ):
        delta_x = np.interp( x, x_from, delta)
        return x + delta_x


    def get_data_with_detph( self, name, warped_depth=False, apply_drops=False ):
        alias = {'Fr':'Fr_','Rf':'Rf_','Bq':'Bq_','Qt':'Qt_'} #normalized stored w. aliases as 'Qt' and 'qt' are the same in windows
        if name in alias.keys():
            name=alias[name]

        # get raw data
        if hasattr( self, 'log_file'):
            d = self.log_file.cpt_data[name]['d']
            data = self.log_file.cpt_data[name]['value']
        else:
            d = self.data['d']
            data = self.data[name]

        if warped_depth:
            d=self.warp_depths( d )

        if apply_drops and hasattr( self, 'drops' ):
            drops = self.drops
            if not warped_depth: # warp drops back to original coords
                dr_from = [d[0] for d in drops]
                dr_to = [d[1] for d in drops]

                dr_from=self.warp_depths( dr_from, inverse_warp=True )
                dr_to=self.warp_depths( dr_to, inverse_warp=True )

                drops = [] # overwrite
                for df, dt in zip( dr_from, dr_to ):
                    drops.append( (df,dt) )

            d, data = self.apply_drops( d, data, drops )

        return d, data


    def apply_drops( self, depth, data, drops ):
        tmp_depth = depth.copy()
        tmp_data = data.copy()

        for d in drops:
            filter_from = depth>=d[0]
            filter_to = depth<=d[1]
            mask = np.multiply(filter_from, filter_to)

            tmp_depth[mask]=np.nan
            tmp_data[mask]=np.nan

        return tmp_depth, tmp_data


    def add_coordinates( self, coords ):
        self.UTM32E_X = float( coords['UTM32-E'] )
        self.UTM32N_Y = float( coords['UTM32-N'] )
        self.NN2000_Z = float( coords['NN2000-Z'] )
        self.test_description = coords['DESCRIPTION']
        self.target_rate = self.test_description.split('Rate:')[1].strip()


    def get_coordinates( self ):
        if hasattr( self, 'UTM32E_X'):
            return self.UTM32E_X, self.UTM32N_Y, self.NN2000_Z
        else:
            raise Exception('error: no coordinates are set for sounding ' + self.pos_name)


    def read_file( self, filename ):
        with open( filename , "r" ) as myfile:
            file_contents = myfile.read()

        return file_contents


    def replace_unwanted_chars( self, raw_file ):
        raw_file = raw_file.replace( '\x00', '' ) # null
        return raw_file


    def split_raw_file( self, raw_file ):
        #if '#$' in raw_file: soundings= raw_file.split( '#$' )
        #else: soundings= raw_file.split( '$\n' )
        soundings= raw_file.split( '$\n' )

        for s in soundings:
            should_break=False
            if 'hm=07,' in s.lower() or 'hm=7,' in s.lower() or 'hm=7\n':
                n_lines = len(s)-len(s.replace('\n',''))
                n_q = max(len(s)-len(s.lower().replace('q=','')) , len(s)-len(s.lower().replace('qc=','')))
                if n_q > n_lines*0.95: should_break=True
            if should_break: break
        if should_break==False: 
            self.remove_sounding=True # no sounding matched criteria
            return None, None

         # trim away everything after first sounding
        if '$\n' in s: s = s.split( '$\n' )[1] # trim to start of headers
        return s.split( '#\n' ) # tuple: (headers, data)


    def check_for_log_file( self ):
        log_filename = self.filename.replace('.cpt', '.log')
        if os.path.isfile( log_filename ):
            return log_filename
        return None


    def get_posname( self, file_path ):
        '''Extracts filename from path. Returns it without extension'''
        filename = os.path.split( file_path )[-1]
        return filename[:-4]


    def parse_headers( self ):
        self.headers = {}

        replacements = [[' HN=',' ,HN='], [',,',','], ['$',''], ['\n','']]
        for r in replacements:
            self.raw_header = self.raw_header.replace(r[0],r[1])
        headers = self.raw_header.split(',') # access individual headers

        for header in headers:
            if '=' in header: # ignore headers created by user using ',' in freetext
                header.replace('=',' = ') # some headers can be empty
                split_header = header.split('=')
                self.headers[split_header[0].strip()] = split_header[1].strip()
        # all variables to lower case
        self.headers = { k.lower() if isinstance( k, str ) else k:v for k,v in self.headers.items()}


    def update( self ):
        # update stresses
        self.data['sig_v0'] = self.soil_stress_model.calc_sigma_v0( self.data['d'] )
        self.data['u0'] = self.soil_stress_model.calc_u0( self.data['d'] )
        self.data['sig_v0_eff'] = self.soil_stress_model.calc_sigma_v0_eff( self.data['d'] )

        # cpt specific
        self.calc_cpt_params()
        self.calc_lowess_params()


    def parse_data( self ):
        ## inputs defined as ( stored in lowercase! )
        # D     depth meters
        # QC    tip resistance MPa (we convert to kPa)
        # FS    sleeve frictional resistance kPa
        # U     u2 pore pressure kPa
        # %     time of registration YYYY-MM-DD-HH-MM-SS-milliseconds, or  (without the "-")
        # TA    tilt angle deg
        # O     temperature deg
        # M     conductivity siemens/m
        # A     push resistance kN
        # B     feed rate mm/s

        if 'hn' in self.headers:
            if not self.headers['hn'].strip().isnumeric(): # cone number not a number (missing ID)
                self.remove_sounding = True
                self.removal_reason = 'cone number not recognized (' + self.headers['hn'] + ')'
                return

        data_headers = ['D', 'QC', 'FS', 'U', '%', 'TA', 'B', 'O', 'M', 'A'] # missing headers ind data removed
        replacements = [ # transform ENVI headers to Geotech
            ['%','%='],
            ['F=','FS='],
            ['Q=','QC='],
            ['IB=','TA='],
            ['\n,',','] # dataline split over two or more lines
        ]

        if replacements:
            for r in replacements:
                self.raw_data = self.raw_data.replace(r[0],r[1])

        # check if data is present
        n_threshold = int(((len(self.raw_data)-len(self.raw_data.replace('D=', '') ))/2) * 0.999)
        data_headers = [header for header in data_headers if ((len(self.raw_data)-len(self.raw_data.replace(header+'=', '')))/2)>n_threshold]

        # create datastructure
        self.data = {}
        for header in data_headers:
            if str( ',' + header + '=' ) in str( ',' + self.raw_data ):
                self.data[header] = np.array([], dtype=np.float32)

        lines = self.raw_data.split('\n')
        for line in lines:
            used_keys = [] # reset
            if '=' in line:
                line = ' ' + line + ', '
                for key in self.data:
                    if key not in used_keys: # only first registration
                        val = line.split(key+'=')[1].split(',')[0]
                        val = float(val) if val else np.nan                        
                        self.data[key] = np.append(self.data[key], val )
                        used_keys.append( key )


        if len(self.data)==0: return # exit for no data

        self.data['QC'] *= 1000 # tip resistance to kPa

        self.data = { k.lower() if isinstance( k, str ) else k:v for k,v in self.data.items()} # vars to lower case


    def calc_cpt_params( self ):
        # set cone correction factors
        alfa = self.calibration['alfa'] if 'alfa' in self.calibration else -1
        beta = self.calibration['beta'] if 'beta' in self.calibration else -1

        # should not be needed!
        if alfa == -1 and 'ma' in self.headers:
            alfa = float( self.headers['ma'] )
        elif alfa == -1:
            alfa = 1
            print( 'No tip calibration found for cone (certificate or file). Using alfa=1')
        if beta == -1 and 'mb' in self.headers:
            beta = float( self.headers['mb'] )
        elif beta == -1:
            beta = 0
            print( 'No sleeve calibration found for cone (certificate or file). Using beta=0')

        self.data['du'] = self.data['u'] - self.data['u0']
        self.data['qt'] = self.data['qc'] + ( 1-alfa ) * self.data['u']
        self.data['ft'] = self.data['fs'] - ( beta*self.data['u'] + 0.3*self.data['du']*((1-alfa)/15-beta) ) # formula from sgi-i15
        self.data['qn'] = self.data['qt'] - self.data['sig_v0']
        self.data['qe'] = self.data['qt'] - self.data['u']
        self.data['du_sig_v0_eff'] = self.data['du'] / self.data['sig_v0_eff']

        # noramlized ("additional '_' for unique filename")
        self.data['Qt_'] = self.data['qn'] / self.data['sig_v0_eff']
        self.data['Bq_'] = self.data['du'] / self.data['qn']
        self.data['Fr_'] = self.data['fs'] / self.data['qn'] * 100
        self.data['Rf_'] = self.data['fs'] / self.data['qt'] * 100

        # removed: not used and data caused problems with log10(num) where num<=0
        #self.data['Ic'] = ((3.47-np.log10(self.data['Qt']))**2 + (np.log10(self.data['Fr'])+1.22)**2)**0.5

    def calc_lowess_params( self ):
        smoother = lowess( delta=self.lowess_interval )
        
        d = self.data['d']
        res = {}

        for param in self.data:
            if param=='d': continue
            smoother.fit( d, self.data[ param ] )
            res[ 'lowess_' + param ] = smoother.predict()
            res[ 'res_' + param ] = self.data[param] - res[ 'lowess_' + param ]
        self.data.update( res )
            


    def geometrical_depth_correction( self, adjust_offset=True ):
        offset_id = 'offset_'
        all_calibration_keys = list( self.calibration.keys() )
        offset_keys = [ k for k in all_calibration_keys if offset_id in k ]

        d_increments = np.round(np.diff( self.data['d'] ),3) # round increments to mm

        values, counts = np.unique(d_increments, return_counts=True) #find frequency of each increment
        base_increment= values[counts.argmax()] # get most frequent increment

        for offset_key in offset_keys: # compute and apply depth correction for each parameter
            param = offset_key.split('_')[1]
            param_desired_offset = self.calibration[offset_key] # get from calibration

            param_calc_shift = int( param_desired_offset // base_increment ) # calc closest corresponding shift
            param_calc_offset = param_calc_shift * base_increment # corresponding offset (can differ a little from desired)

            if adjust_offset: self.calibration[offset_key] = param_calc_offset # overwrite calibration values

            data_w_offset = self.data[param][ param_calc_shift: ] # shift (cuts end)
            offset_padding = np.empty((param_calc_shift,)) # create end padding for shift
            offset_padding[:] = np.nan
            self.data[param] = np.append( data_w_offset, offset_padding ) # replace data with shifted and padded values


    def data_out( self ):
        d_out = np.arange( 0, 20.01, 0.02 )
        d = self.data['d']
        sm = self.soil_stress_model
        d_sm = sm.d

        q_out = np.interp( d_out, d, self.data['qc'] )
        f_out = np.interp( d_out, d, self.data['fs'] )
        u_out = np.interp( d_out, d, self.data['u'] )

        sig_out = np.interp( d_out, sm.d, sm.sigma_v0 )
        u0_out = np.interp( d_out, sm.d, sm.u0 )


        import sys
        np.set_printoptions(threshold=sys.maxsize)
        if False:
            print( 'd:', d_out )
            print( 'q:', q_out )
            print( 'f:', f_out )
            print( 'u:', u_out )
        elif False:
            print( 'd:', d_out )
            print( 'sig:', sig_out )
            print( 'u0:', u0_out )

        a=1

    def plot( self, plot_cpt_data=True, plot_log_data=True, marker=None, fontsize=None, plot_title=False, tick_f_size=None ):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        if fontsize==None: fontsize = 10
        if tick_f_size==None: tick_f_size = 10
        if marker==None: marker='.'
        over_all_scale = 0.5
        fig_width = 25 * over_all_scale
        fig_height = (1 + (self.data['d'][-1] // 5) * 5 ) * over_all_scale # new increment each 5m

        fig, axs = plt.subplots( 1, 3, sharey=True, figsize=(fig_width, fig_height), gridspec_kw={'width_ratios': [ 2, 1, 2 ]} )

        ps = 5 # point size 0-no points
        w = .8

        c_blue  = ( 0/255,142/255,194/255 )
        c_red   = ( 237/255,28/255,46/255 )
        c_water = ( 0, 0, 0 )

        if plot_cpt_data:
            ref =  self.data
            if 'qc' in ref:
                #axs[0].plot( ref['qc'], ref['d'], ls='-', marker=marker, markersize=ps, lw=w, color=c_blue, label=r'$q_c$' )
                axs[0].plot( ref['qt'], ref['d'], ls='-', marker=marker, markersize=ps, lw=w, color=c_blue, label='Data file' )
            if 'fs' in ref:
                axs[1].plot( ref['fs'], ref['d'], ls='-', marker=marker, markersize=ps, lw=w,  color=c_blue, label='cpt-file' )
            if 'u' in ref:
                axs[2].plot( ref['u'], ref['d'], ls='-', marker=marker, markersize=ps, lw=w,  color=c_blue, label='cpt-file', zorder=2)
                #axs[2].plot( self.soil_stress_model.u0, self.soil_stress_model.d, ls='--', marker=marker, markersize=ps, lw=1,  color=c_water, label=r'$u_0$' )

        if plot_log_data and hasattr(self, 'log_file'):
            ref = self.log_file.cpt_data
            if 'qc' in ref:
                axs[0].plot( ref['qt']['value'], ref['qc']['d'], ls='-', marker=marker, markersize=ps, lw=w, color=c_red, label='Log file' )
            if 'fs' in ref:
                axs[1].plot( ref['fs']['value'], ref['fs']['d'], ls='-', marker=marker, markersize=ps, lw=w, color=c_red, label='log-file' )
            if 'u' in ref:
                axs[2].plot( ref['u']['value'], ref['u']['d'], ls='-', marker=marker, markersize=ps, lw=w, color=c_red, label='log-file' )

        self.format_axes( axs, fontsize, tick_f_size )
        if plot_log_data: self.draw_breaks( axs, Rectangle )

        if plot_title: fig.suptitle( self.pos_name )

        axs[0].legend( fontsize=tick_f_size )
        #axs[2].legend()
        plt.tight_layout( w_pad=3 )
        plt.show()


    def format_axes( self, axs, fontsize, tick_f_size ):
        min_d, max_d = self.min_max_depth( self.data['d'] )

        axs[0].set_ylim( min_d, max_d )
        axs[0].invert_yaxis()
        axs[0].set_ylabel('Depth (m)', fontsize=fontsize )
        axs[0].set_xlabel( 'q' + r'$_t$' + ' (kPa)', fontsize=fontsize )
        axs[1].set_xlabel( 'f' + r'$_s$' + ' (kPa)', fontsize=fontsize )
        axs[2].set_xlabel( 'u' + r'$_2$' + ' (kPa)', fontsize=fontsize )
        for ax in axs:
            ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, labelsize=tick_f_size)
            ax.xaxis.set_label_position('top')
            x_min, x_max = self.min_max_val( ax )
            ax.set_xlim( x_min, x_max )            
            #ax.spines['bottom'].set_visible(False)
            #ax.spines['right'].set_visible(False)
            ax.grid(True)
        
        if False:
            axs[0].set_xlim(500,1700)
            axs[1].set_xlim(0,20)
            axs[2].set_xlim(50,300)
            axs[0].set_ylim( 10.4, 9.9 )


    def min_max_depth( self, depth_array ):
        y_min = 0
        y_max = (( depth_array[-1] // 5 ) + 1 ) * 5
        return y_min, y_max


    def min_max_val( self, ax ):
        cur_min, cur_max = ax.get_xlim()
        ret_min = 0 if abs(cur_min) < cur_max else cur_min #and cur_min>0 * .1
        n = int(np.log10(cur_max))
        ret_max = 10**n * ( (cur_max//10**n) + 1 )
        return ret_min, ret_max


    def draw_breaks( self, axs, Rectangle ):
        for ax in axs:
            x_min, x_max = ax.get_xlim()
            w = x_max - x_min
            h = 0.07
            delta = h/2

            if hasattr(self, 'log_file'):
                for some_break in self.log_file.breaks:
                    shifted_break = some_break+delta
                    y_ref = shifted_break-delta
                    ax.add_patch( Rectangle( (x_min, y_ref), w, h, facecolor = (0/255,142/255,194/255, 0.5) ) )