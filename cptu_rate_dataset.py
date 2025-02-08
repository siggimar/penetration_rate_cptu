import os
import pickle
import numpy as np
from tqdm import tqdm # progressbar while reading files
from soil_stresses import soil_stress_model
from cpt import cpt


# class to import soundings and establish a soil stress profile for each site

class dataset():
    '''
    Access data from cptu-tests
    ---------------------------
    load data via load_dataset method

    NGTS: Tiller-Flotten (dataset 0)
     0  TILC44      PD:  4.00m.  CPTU:  4.00m-20.02m.  Rate: 15mm/s
     1  TILC50      PD:  4.00m.  CPTU:  4.00m-20.06m.  Rate: 65mm/s
     2  TILC51      PD:  4.00m.  CPTU:  4.00m-20.06m.  Rate: 50mm/s
     3  TILC52      PD:  4.00m.  CPTU:  4.00m-20.18m.  Rate: 50mm/s
     4  TILC53      PD:  4.00m.  CPTU:  4.00m-20.04m.  Rate: 30mm/s
     5  TILC54      PD:  4.00m.  CPTU:  4.00m-20.04m.  Rate: 25mm/s
     6  TILC55      PD:  4.00m.  CPTU:  4.00m-20.02m.  Rate: 20mm/s
     7  TILC57      PD:  4.00m.  CPTU:  4.00m-20.02m.  Rate: 20mm/s (Reference sounding)
     8  TILC58      PD:  4.00m.  CPTU:  4.00m-20.08m.  Rate: 65mm/s
     9  TILC59      PD:  4.00m.  CPTU:  4.00m-20.00m.  Rate: 10mm/s
    10  TILC65      PD:  4.00m.  CPTU:  4.00m-20.04m.  Rate: 20mm/s
    11  TILC66      PD:  4.00m.  CPTU:  4.00m-20.02m.  Rate: 20mm/s
    12  TILC69      PD:  4.00m.  CPTU:  4.00m-20.00m.  Rate:  5mm/s
    13  TILC72      PD:  4.00m.  CPTU:  4.00m-20.04m.  Rate: 65mm/s
    14  TILC74      PD:  4.00m.  CPTU:  4.00m-20.02m.  Rate: 15mm/s
    15  TILC75      PD:  4.00m.  CPTU:  4.00m-20.00m.  Rate:  5mm/s
    16  TILC77      PD:  4.00m.  CPTU:  4.00m-20.10m.  Rate: 65mm/s
    17  TILC78      PD:  4.00m.  CPTU:  4.00m-20.04m.  Rate: 50mm/s
    18  TILC79      PD:  4.00m.  CPTU:  4.00m-20.04m.  Rate: 30mm/s
    19  TILC81      PD:  4.00m.  CPTU:  4.00m-20.02m.  Rate: 10mm/s
    20  TILC85      PD:  4.00m.  CPTU:  4.00m-20.02m.  Rate: 20mm/s
    21  TILC87      PD:  4.00m.  CPTU:  4.00m-20.20m.  Rate: 35mm/s
    22  TILC88      PD:  4.00m.  CPTU:  4.00m-20.10m.  Rate: 25mm/s
    23  TILC89      PD:  4.00m.  CPTU:  4.00m-20.06m.  Rate: 35mm/s
    24  TILC90      PD:  4.00m.  CPTU:  4.00m-20.06m.  Rate: 50mm/s

    NGTS: Øysand (dataset 1)
     0  OYSC05      PD:  8.00m.  CPTU:  8.00m-20.40m.  Rate: 15mm/s
     1  OYSC06      PD:  8.00m.  CPTU:  8.00m-19.72m.  Rate: 35mm/s
     2  OYSC11      PD:  8.00m.  CPTU:  8.00m-21.66m.  Rate: 30mm/s
     3  OYSC15      PD:  8.00m.  CPTU:  8.00m-17.02m.  Rate: 30mm/s
     4  OYSC19      PD:  8.00m.  CPTU:  8.00m-18.34m.  Rate: 20mm/s (Reference sounding)
     5  OYSC33      PD:  8.00m.  CPTU:  8.00m-19.70m.  Rate: 20mm/s
     6  OYSC47      PD:  8.00m.  CPTU:  8.00m-17.62m.  Rate: 50mm/s
     7  OYSC48      PD:  8.00m.  CPTU:  8.00m-18.76m.  Rate: 50mm/s
     8  OYSC54      PD:  8.00m.  CPTU:  8.00m-22.46m.  Rate: 65mm/s
     9  OYSC63      PD:  8.00m.  CPTU:  8.00m-16.78m.  Rate: 65mm/s
    10  OYSC64_1    PD:  8.00m.  CPTU:  8.00m-13.62m.  Rate: 20mm/s
    11  OYSC64_2    PD: 15.00m.  CPTU: 15.00m-18.34m.  Rate: 20mm/s
    12  OYSC73      PD:  8.00m.  CPTU:  8.00m-17.18m.  Rate: 15mm/s
    13  OYSC76      PD:  8.00m.  CPTU:  8.00m-20.22m.  Rate: 25mm/s
    14  OYSC78      PD:  8.00m.  CPTU:  8.00m-18.42m.  Rate: 35mm/s
    15  OYSC80      PD:  8.00m.  CPTU:  8.00m-23.46m.  Rate: 50mm/s
    16  OYSC82      PD:  8.00m.  CPTU:  8.00m-17.36m.  Rate: 65mm/s
    17  OYSC83      PD:  8.00m.  CPTU:  8.00m-16.98m.  Rate:  5mm/s
    18  OYSC84      PD:  8.00m.  CPTU:  8.00m-17.64m.  Rate: 25mm/s
    19  OYSC85      PD:  8.00m.  CPTU:  8.00m-16.56m.  Rate:  5mm/s
    20  OYSC87      PD:  8.00m.  CPTU:  8.00m-20.08m.  Rate: 10mm/s
    21  OYSC88      PD:  8.00m.  CPTU:  8.00m-16.18m.  Rate: 10mm/s
    22  OYSC89      PD:  8.00m.  CPTU:  8.00m-21.94m.  Rate: 65mm/s
    23  OYSC90      PD:  8.00m.  CPTU:  8.00m-15.76m.  Rate: 20mm/s
    24  OYSC92      PD:  8.00m.  CPTU:  8.00m-15.38m.  Rate: 20mm/s
    25  OYSC93      PD:  8.00m.  CPTU:  8.00m-20.68m.  Rate: 50mm/s

    Stjørdal: Halsen (dataset 2)
     0  HALS01      PD:  3.00m.  CPTU:  3.00m-19.81m.  Rate: 20mm/s
     1  HALS02      PD:  3.00m.  CPTU:  3.00m-19.76m.  Rate: 20mm/s
     2  HALS03      PD:  3.00m.  CPTU:  3.00m-19.81m.  Rate: 20mm/s
     3  HALS04      PD:  3.00m.  CPTU:  3.00m-19.80m.  Rate: 20mm/s
     4  HALS05      PD:  3.00m.  CPTU:  3.00m-19.81m.  Rate: 20mm/s (Reference sounding)
     5  HALS06      PD:  3.00m.  CPTU:  3.00m-19.78m.  Rate: 10mm/s
     6  HALS07      PD:  3.00m.  CPTU:  3.00m-19.74m.  Rate: 40mm/s
     7  HALS08      PD:  3.00m.  CPTU:  3.00m-19.76m.  Rate: 30mm/s
     8  HALS09      PD:  3.00m.  CPTU:  3.00m-19.78m.  Rate: 15mm/s
     9  HALS10      PD:  3.00m.  CPTU:  3.00m-19.77m.  Rate: 30mm/s
    10  HALS11      PD:  3.00m.  CPTU:  3.00m-19.79m.  Rate: 15mm/s
    11  HALS12      PD:  3.00m.  CPTU:  3.00m-19.79m.  Rate: 10mm/s
    12  HALS13      PD:  3.00m.  CPTU:  3.00m-19.81m.  Rate: 40mm/s
    '''


    def __init__( self ):
        self.saves_folder = 'saves'
        self.clear_dataset() # init

        self.rate_colors = { # for consistent plotting
            '5mm/s' : (0,176,240),
            '10mm/s': (112,48,160),
            '15mm/s': (255,0,255),
            '20mm/s': (192,0,0),
            '25mm/s': (255,150,0),
            '30mm/s': (146,208,80),
            '35mm/s': (153,102,51),
            '40mm/s': (0,128,128),
            '50mm/s': (0,102,255),
            '65mm/s': (0,0,0),
        }

        self.rate_colors =  { key: (r/255, g/255, b/255) for key, (r, g, b) in self.rate_colors.items() }


    def clear_dataset( self ):
        self.soundings  = []
        self.reference_sounding = None


    def load_dataset( self, location=0, borehole=-1, read_logfiles=False, from_file='', txt_overview=False ):
        '''
        populates soundings list with CPT objects.

        loads from file if from_file=True and a dataset.pkl file exists
        '''

        base_dir = 'data'

        # available subfolders with data
        folders = {
                0: '0_tiller_flotten',
                1: '1_oysand',
                2: '2_stjordal'
            }

        self.reference_name = {0:'TILC57',1:'OYSC19',2:'HALS05'}[location]
        self.location_name = {0:'NGTS: Tiller-Flotten',1:'NGTS: Øysand',2:'Stjørdal: Halsen'}[location]
        self.txt_overview = txt_overview # generates table over all soundings at position

        self.location = location
        from_file = str( self.location ) + '_' + from_file

# data from site characterization reports or relevant papers
        # soil weight profile 
        gamma = {  # id [ [depts], [gammas] ]
            0: [ [1.82,2.60,3.40,4.20,5.00,5.80,6.60,7.40,8.20,9.20,10.40,
                  11.25,12.30,13.15,14.50,15.40,16.10,17.15,18.00,19.30,19.80], 
                 [18.1,18.0,17.4,17.5,16.8,17.2,16.8,17.2,17.3,17.8,17.8,
                   18.7,17.4,18.0,18.0,18.0,18.0,18.7,18.1,18.4,19.3] ], # NGTS site characterization

            1: [ [1.01, 1.63, 1.99, 2.65, 3, 3.48, 5.65, 5.99, 6.5, 7, 7.62,
                   9.01, 9.61, 10.01, 10.61, 11, 11.62, 12.01, 12.64, 13.01, 13.63, 14.01,
                     14.62, 15.02, 15.61, 16, 16.6, 17, 17.6, 18.02, 18.65, 19, 19.69],
                 [16.96, 16.96, 16.83, 16.83, 14.88, 14.88, 20.17, 20.25, 20.25, 19.38,
                   19.38, 19.63, 19.63, 19.46, 19.46, 19, 19.04, 18.96, 18.96,
                     18.42, 18.38, 18.75, 18.75, 18.38, 18.38, 18.38, 18.38, 19.29, 19.29,
                       18.54, 18.58, 17.25, 17.25] ], # NGTS site characterization (2 registrations of 13kN/m3 at 8& 8.6m removed)

            2: [ [4.05, 4.05, 5.05, 6.04, 6.06, 7.04, 7.08, 7.08, 7.09, 8.06, 
                  8.08, 9.05, 9.08, 10.05, 10.10, 11.04, 11.07, 12.05, 13.05],
                 [20.7, 21.5, 22.7, 18.1, 20.3, 20.8, 20.7, 19.3, 20.7, 20.2,
                   19.6, 21.1, 22.3, 20.7, 20.7, 22.0, 19.1, 20.6, 20.9]], # Bihs 2021
        }

        # porepressure profile
        u0 = { # id : [ [depts], [u0s] ]
            0: [ [0.00,1.50,5.00,7.00,15.75,22.90], [0.0,0.0,30.0,36.0,56.0,68.0] ], # u_0 profile
            1: [ [2], [0] ], # only define groundwater level
            2: [ [1.5], [0] ], # GWL
        }

        self.c_v = { # best estimate from available data in mm^2/s
            0: [ [2.5, 4, 17, 20], [0.5074, 0.2220, 0.6342, 1.7440] ], # from CRS + CPTu correlations ( site characterization report: SCR)
            1: [ [2], [950] ], # no estimate in litterature:  using 30000m2/yr
            2: [ [ 2.5, 7.8, 8.0, 8.2, 9.0, 12.6, 14.5, 15.0, 15.2, 16.0 ], # from SCR:  CRS Oedometer, but depth-adjusted to match CPTu profile
                [ 23.148, 23.148, 104.642, 17.440, 15.855, 1.903, 19.026, 19.026, 82.445, 82.445 ] ], 
        }[location] # keep current


        # calculation units
        self.unit_defs = {
            0: [ [7.5, 8.2], [8.6, 9.8], [10.1, 11.1], [11.4, 12.8 ], [13, 14.5], [14.8, 16.4], [17.2, 18.0], [18.0, 19.0], [19.1, 19.8] ], #[6.80, 7.25], 
            1: [ [8.8, 9.5], [9.5, 9.9], [10, 10.66], [10.9, 11.55], [11.6, 12.33], [12.4, 12.93], [13.0, 13.75], [14.1, 14.65], [16.0,16.7] ],
            2: [ [3.9, 4.7], [4.75, 5.55], [5.6, 6.3], [6.75,7.75], [8.4, 9.5], [10.15, 12.15], [12.82, 13.92], [14.5, 15.25],  ]
        }[location] # keep current



        # set working folder
        load_folder = os.path.join( base_dir, folders[location] )

        files = self.get_files_in_folder( load_folder )
        cpts = [f for f in files if '.cpt' in f.lower()]
        coords = self.read_coord_file( load_folder )

        if borehole>-1: # a selected sounding?
            cpts = [ cpts[borehole] ]

        print( '\nbuilding soil stress profile.')
        local_stress_model = soil_stress_model( gamma[ location ], u0[ location ] )

        print( '\n\nreading sounding data...')

        from_file_path= os.path.join( self.saves_folder, from_file )
        if from_file != '' and os.path.isfile( from_file_path ):
            with open( from_file_path, 'rb' ) as f:
                self.soundings = pickle.load( f )
                print('pre-processed soundings read from file')
        else:
            for sounding in tqdm(range(len(cpts))):
                self.soundings.append( 
                    cpt( cpts[ sounding ], soil_stress_model=local_stress_model, read_log_file=read_logfiles )
                )
                self.soundings[-1].add_coordinates( coords[ self.soundings[-1].pos_name ] )

        if self.txt_overview: self.overview()


    def save_to_file( self, filename='dataset.pkl' ):
        os.makedirs( self.saves_folder, exist_ok=True )
        if filename == '': return

        filename = str(self.location) + '_' + filename
        file_path = os.path.join( self.saves_folder, filename )

        with open( file_path, 'wb' ) as f:
            pickle.dump( self.soundings, f )


    def get_files_in_folder( self, folder ):
        files = []
        if os.path.isdir( folder ):
            files = [os.path.join( folder, f ) for f in os.listdir( folder ) if os.path.isfile( os.path.join(folder, f) )]
        return files


    def read_coord_file( self, folder ):
        f = '_coords.csv'
        filename = os.path.join( folder, f )

        with open( filename , "r" ) as myfile:
            file_contents = myfile.read()

        lines = file_contents.split('\n')
        headers = lines[0].split(',')
        coords = {}

        for i in range( 1, len(lines) ): # exclude headers
            tmp_dict = {}
            line = lines[i].split(',')
            for i in range(5):
                tmp_dict[ headers[i] ] = line[i]
            coords[line[0]] = tmp_dict

        return coords


    def sounding_index_by_name( self, sounding_name ):
        for i in range( len(self.soundings) ):
            if self.soundings[i].pos_name == sounding_name:
                return i
        return -1


    def all_pos_names( self ):
        return [ s.pos_name for s in self.soundings ]


    def sounding_by_name( self, sounding_name):
        idx = self.sounding_index_by_name( sounding_name )
        return self.soundings[idx]


    def get_reference( self ):
        return self.soundings[ self.sounding_index_by_name( self.reference_name ) ]


    def rates_freqs( self ):
        # prints out statistics on rates and registration frequencies in dataset
        column_width = 12
        decimals = 5

        rates = {}
        freqs = {}

        for s in self.soundings:
            b = s.data['b']
            b_log = s.log_file.rate['avg_rate']
            std_log = s.log_file.rate['rate_st_dev']
            d = s.data['d']
            rates[s.pos_name] = [ np.average(b), b_log, np.std(b), std_log, d[0], d[-1]]

            cone_nr = s.calibration['cone_number']
            s_freqs = s.log_file.avg_log_freq
            if cone_nr in freqs:
                freqs[cone_nr].append(s_freqs)
            else:
                freqs[cone_nr] = [s_freqs]
        print( 'penetration rates' )


        print('Frequencies: ')
        some_header = 'Cone number'.ljust(14) 
        some_header += 'avg(q_c)'.ljust(column_width) + 'std(q_c)'.ljust(column_width)
        some_header += 'avg(f_s)'.ljust(column_width) + 'std(f_s)'.ljust(column_width)
        some_header += 'avg(u_2)'.ljust(column_width) + 'std(u_2)'.ljust(column_width)
        print( some_header )
        for c_n in freqs:
            qc=np.array([])
            fs=np.array([])
            u2=np.array([])
            for items in freqs[ c_n ]:
                qc = np.append(qc,  items['qc'] )
                fs = np.append(fs,  items['fs'] )
                u2 = np.append(u2, items['u'] )
            # pretty print a summary
            out_ln = str(c_n).ljust(14)
            out_ln += str( np.round( np.average(qc), decimals) ).ljust(column_width)
            out_ln += str( np.round( np.std(qc), decimals) ).ljust(column_width)
            out_ln += str( np.round( np.average(fs), decimals) ).ljust(column_width)
            out_ln += str( np.round( np.std(fs), decimals) ).ljust(column_width)
            out_ln += str( np.round( np.average(u2), decimals) ).ljust(column_width)
            out_ln += str( np.round( np.std(u2), decimals) ).ljust(column_width)


            print( out_ln )
        a=1


    def overview( self ):
        print( self.location_name  + ' (dataset ' + str(self.location) + ')' )
        for i, s in enumerate(self.soundings):
            s_rate = s.test_description.split('Rate:')[1].strip().rjust(6)
            d_from = '{:.2f}'.format( round(s.data['d'][0], 2) ).rjust(5)
            d_to = '{:.2f}'.format( round(s.data['d'][-1], 2) ).rjust(5)

            out_ln = str(i).rjust(6) + '  '
            out_ln += s.pos_name.ljust( 12 )
            out_ln += 'PD: ' + d_from + 'm.  '
            out_ln += 'CPTU: ' + d_from + 'm-' + d_to + 'm.  '
            out_ln += 'Rate: ' + s_rate

            print( out_ln )