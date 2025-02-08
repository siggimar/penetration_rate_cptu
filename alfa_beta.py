'''
    Script contains calibration data for cones in study.
    the class calibration_data is initialized with all calibrations, which are read in as dicts
    
    the class method get_sertificate( CONE_NUMBER, DATE ) returns the active calibration on provided date
'''
from datetime import datetime

cptu_calibrations = {
    4364:{
        '21-12-2010' : {'date' : '21-12-2010', 'alfa' : 0.810, 'max_qc' : 50, 'scale_fact_qc' : 1267, 'res_qc' : 0.6022, 'temp_eff_qc' : 19.2704, 'beta' : 0,     'max_fs' : 0.5, 'scale_fact_fs' : 3615, 'res_fs' : 0.0106, 'temp_eff_fs' : 0.3604, 'max_u' : 2, 'scale_fact_u' : 3798, 'res_u' : 0.0201, 'temp_eff_u' : 0.5427, 'temp_scale_min' : 0, 'temp_scale_max' : 40},
        '06-09-2012' : {'date' : '06-09-2012', 'alfa' : 0.838, 'max_qc' : 50, 'scale_fact_qc' : 1268, 'res_qc' : 0.6017, 'temp_eff_qc' : 77.6193, 'beta' : 0,     'max_fs' : 0.5, 'scale_fact_fs' : 3555, 'res_fs' : 0.0107, 'temp_eff_fs' : 1.4873, 'max_u' : 2, 'scale_fact_u' : 3788, 'res_u' : 0.0201, 'temp_eff_u' : 0.5628, 'temp_scale_min' : 0, 'temp_scale_max' : 40},
        '07-11-2013' : {'date' : '07-11-2013', 'alfa' : 0.842, 'max_qc' : 50, 'scale_fact_qc' : 1277, 'res_qc' : 0.5974, 'temp_eff_qc' : 30.4674, 'beta' : 0,     'max_fs' : 0.5, 'scale_fact_fs' : 3634, 'res_fs' : 0.0105, 'temp_eff_fs' : 0.7455, 'max_u' : 2, 'scale_fact_u' : 3788, 'res_u' : 0.0201, 'temp_eff_u' : 0.7035, 'temp_scale_min' : 0, 'temp_scale_max' : 40},
        '03-07-2015' : {'date' : '03-07-2015', 'alfa' : 0.839, 'max_qc' : 50, 'scale_fact_qc' : 1278, 'res_qc' : 0.5970, 'temp_eff_qc' : 22.6860, 'beta' : 0,     'max_fs' : 0.5, 'scale_fact_fs' : 3648, 'res_fs' : 0.0104, 'temp_eff_fs' : 0.4368, 'max_u' : 2, 'scale_fact_u' : 3778, 'res_u' : 0.0202, 'temp_eff_u' : 0.8080, 'temp_scale_min' : 0, 'temp_scale_max' : 40},
        '22-10-2018' : {'date' : '22-10-2018', 'alfa' : 0.860, 'max_qc' : 50, 'scale_fact_qc' : 1275, 'res_qc' : 0.5984, 'temp_eff_qc' : 22.7250, 'beta' : 0,     'max_fs' : 0.5, 'scale_fact_fs' : 3651, 'res_fs' : 0.0104, 'temp_eff_fs' : 0.3750, 'max_u' : 2, 'scale_fact_u' : 3769, 'res_u' : 0.0202, 'temp_eff_u' : 0.2830, 'temp_scale_min' : 0, 'temp_scale_max' : 40},
        '22-01-2020' : {'date' : '22-01-2020', 'alfa' : 0.859, 'max_qc' : 50, 'scale_fact_qc' : 1269, 'res_qc' : 0.6012, 'temp_eff_qc' : 10.8150, 'beta' : 0,     'max_fs' : 0.5, 'scale_fact_fs' : 3642, 'res_fs' : 0.0105, 'temp_eff_fs' : 0.2300, 'max_u' : 2, 'scale_fact_u' : 3779, 'res_u' : 0.0202, 'temp_eff_u' : 0.8670, 'temp_scale_min' : 0, 'temp_scale_max' : 40},
        '01-03-2022' : {'date' : '01-03-2022', 'alfa' : 0.873, 'max_qc' : 50, 'scale_fact_qc' : 1268, 'res_qc' : 0.6017, 'temp_eff_qc' : 23.4520, 'beta' : 0,     'max_fs' : 0.5, 'scale_fact_fs' : 3639, 'res_fs' : 0.0105, 'temp_eff_fs' : 0.4920, 'max_u' : 2, 'scale_fact_u' : 3794, 'res_u' : 0.0201, 'temp_eff_u' : 1.8490, 'temp_scale_min' : 5, 'temp_scale_max' : 40},
        '03-11-2022' : {'date' : '03-11-2022', 'alfa' : 0.870, 'max_qc' : 50, 'scale_fact_qc' : 1269, 'res_qc' : 0.6012, 'temp_eff_qc' : 11.4160, 'beta' : 0,     'max_fs' : 0.5, 'scale_fact_fs' : 3641, 'res_fs' : 0.0105, 'temp_eff_fs' : 0.3350, 'max_u' : 2, 'scale_fact_u' : 3794, 'res_u' : 0.0201, 'temp_eff_u' : 0.5020, 'temp_scale_min' : 5, 'temp_scale_max' : 40}
    },
    4458:{
        '18-11-2011' : {'date' : '18-11-2011', 'alfa' : 0.852, 'max_qc' : 50, 'scale_fact_qc' : 1251, 'res_qc' : 0.6099, 'temp_eff_qc' : 31.1049, 'beta' : 0,     'max_fs' : 0.5, 'scale_fact_fs' : 3753, 'res_fs' : 0.0101, 'temp_eff_fs' : 0.4343, 'max_u' : 2, 'scale_fact_u' : 3365, 'res_u' : 0.0227, 'temp_eff_u' : 0.8853, 'temp_scale_min' : 0, 'temp_scale_max' : 40},
        '27-06-2013' : {'date' : '27-06-2013', 'alfa' : 0.820, 'max_qc' : 50, 'scale_fact_qc' : 1271, 'res_qc' : 0.6003, 'temp_eff_qc' : 33.0165, 'beta' : 0,     'max_fs' : 0.5, 'scale_fact_fs' : 3748, 'res_fs' : 0.0102, 'temp_eff_fs' : 0.6120, 'max_u' : 2, 'scale_fact_u' : 3369, 'res_u' : 0.0226, 'temp_eff_u' : 0.3616, 'temp_scale_min' : 0, 'temp_scale_max' : 40},        
        '16-04-2014' : {'date' : '16-04-2014', 'alfa' : 0.838, 'max_qc' : 50, 'scale_fact_qc' : 1258, 'res_qc' : 0.6065, 'temp_eff_qc' : 48.5200, 'beta' : 0.001, 'max_fs' : 0.5, 'scale_fact_fs' : 3700, 'res_fs' : 0.0103, 'temp_eff_fs' : 0.9476, 'max_u' : 2, 'scale_fact_u' : 3365, 'res_u' : 0.0227, 'temp_eff_u' : 0.4540, 'temp_scale_min' : 0, 'temp_scale_max' : 40},        
        '09-05-2016' : {'date' : '09-05-2016', 'alfa' : 0.837, 'max_qc' : 50, 'scale_fact_qc' : 1267, 'res_qc' : 0.6022, 'temp_eff_qc' : 39.1180, 'beta' : 0,     'max_fs' : 0.5, 'scale_fact_fs' : 3736, 'res_fs' : 0.0102, 'temp_eff_fs' : 0.5910, 'max_u' : 2, 'scale_fact_u' : 3355, 'res_u' : 0.0227, 'temp_eff_u' : 0.9540, 'temp_scale_min' : 0, 'temp_scale_max' : 40},
        '01-06-2017' : {'date' : '01-06-2017', 'alfa' : 0.854, 'max_qc' : 50, 'scale_fact_qc' : 1266, 'res_qc' : 0.6026, 'temp_eff_qc' : 30.1140, 'beta' : 0,     'max_fs' : 0.5, 'scale_fact_fs' : 3729, 'res_fs' : 0.0102, 'temp_eff_fs' : 0.5620, 'max_u' : 2, 'scale_fact_u' : 3360, 'res_u' : 0.0227, 'temp_eff_u' : 0.6350, 'temp_scale_min' : 0, 'temp_scale_max' : 40},
        '18-11-2019' : {'date' : '18-11-2019', 'alfa' : 0.860, 'max_qc' : 50, 'scale_fact_qc' : 1270, 'res_qc' : 0.6007, 'temp_eff_qc' : 23.4150, 'beta' : 0,     'max_fs' : 0.5, 'scale_fact_fs' : 3731, 'res_fs' : 0.0102, 'temp_eff_fs' : 0.4490, 'max_u' : 2, 'scale_fact_u' : 3364, 'res_u' : 0.0227, 'temp_eff_u' : 0.4530, 'temp_scale_min' : 0, 'temp_scale_max' : 40},
        '01-12-2020' : {'date' : '01-12-2020', 'alfa' : 0.857, 'max_qc' : 50, 'scale_fact_qc' : 1266, 'res_qc' : 0.6026, 'temp_eff_qc' :  7.2270, 'beta' : 0,     'max_fs' : 0.5, 'scale_fact_fs' : 3725, 'res_fs' : 0.0102, 'temp_eff_fs' : 0.1120, 'max_u' : 2, 'scale_fact_u' : 3376, 'res_u' : 0.0226, 'temp_eff_u' : 0.3840, 'temp_scale_min' : 5, 'temp_scale_max' : 40},
        '29-12-2021' : {'date' : '29-12-2021', 'alfa' : 0.865, 'max_qc' : 50, 'scale_fact_qc' : 1267, 'res_qc' : 0.6022, 'temp_eff_qc' : 16.8500, 'beta' : 0,     'max_fs' : 0.5, 'scale_fact_fs' : 3732, 'res_fs' : 0.0102, 'temp_eff_fs' : 0.3880, 'max_u' : 2, 'scale_fact_u' : 3381, 'res_u' : 0.0226, 'temp_eff_u' : 0.6990, 'temp_scale_min' : 0, 'temp_scale_max' : 40},
        '25-11-2022' : {'date' : '25-11-2022', 'alfa' : 0.856, 'max_qc' : 50, 'scale_fact_qc' : 1272, 'res_qc' : 0.5998, 'temp_eff_qc' : 14.9860, 'beta' : 0,     'max_fs' : 0.5, 'scale_fact_fs' : 3793, 'res_fs' : 0.0101, 'temp_eff_fs' : 0.3210, 'max_u' : 2, 'scale_fact_u' : 3654, 'res_u' : 0.0209, 'temp_eff_u' : 1.3770, 'temp_scale_min' : 0, 'temp_scale_max' : 40}        
    },
    4320:{
        '11-10-2013' : {'date' : '11-10-2013', 'alfa' : 0.849, 'max_qc' : 50, 'scale_fact_qc' : 1310, 'res_qc' : 0.5824, 'temp_eff_qc' : 10.4832, 'beta' : 0,     'max_fs' : 0.5, 'scale_fact_fs' : 3653, 'res_fs' : 0.0104, 'temp_eff_fs' : 0.2704, 'max_u' : 2, 'scale_fact_u' : 3642, 'res_u' : 0.0209, 'temp_eff_u' : 0.2930, 'temp_scale_min' : 0, 'temp_scale_max' : 40},
        '20-05-2016' : {'date' : '20-05-2016', 'alfa' : 0.830, 'max_qc' : 50, 'scale_fact_qc' : 1291, 'res_qc' : 0.5910, 'temp_eff_qc' :  7.6780, 'beta' : 0.002, 'max_fs' : 0.5, 'scale_fact_fs' : 3668, 'res_fs' : 0.0104, 'temp_eff_fs' : 0.1030, 'max_u' : 2, 'scale_fact_u' : 3642, 'res_u' : 0.0209, 'temp_eff_u' : 0.2930, 'temp_scale_min' : 0, 'temp_scale_max' : 40},
        '25-09-2019' : {'date' : '25-09-2019', 'alfa' : 0.864, 'max_qc' : 50, 'scale_fact_qc' : 1298, 'res_qc' : 0.5778, 'temp_eff_qc' : 15.2730, 'beta' : 0,     'max_fs' : 0.5, 'scale_fact_fs' : 3679, 'res_fs' : 0.0104, 'temp_eff_fs' : 0.2480, 'max_u' : 2, 'scale_fact_u' : 3661, 'res_u' : 0.0208, 'temp_eff_u' : 0.3740, 'temp_scale_min' : 5, 'temp_scale_max' : 40}
    }
}

cptu_geometry_offset = { # shift from tip to element center in meters. Equal here, but can be specified
    4364:{'offset_qc':0.000,'offset_fs':0.11, 'offset_u':0.04},
    4458:{'offset_qc':0.000,'offset_fs':0.11, 'offset_u':0.04},
    4320:{'offset_qc':0.000,'offset_fs':0.11, 'offset_u':0.04}
}


class calibration_data( ):
    def	__init__( self ):
        self.calibrations = cptu_calibrations
        self.add_shifts() # adds cptu_geometry_offset to all certificates


    def add_shifts( self ):
        for cone_number in cptu_geometry_offset:
            for certificate in self.calibrations[cone_number]:
                self.calibrations[cone_number][certificate].update( cptu_geometry_offset[cone_number] )


    def standard_certificate( self, cone_number, date ):
        # modify most recent certificate from one with most data & return it
        today = datetime.today().strftime('%d-%m-%Y')
        tmp_cone_number = max(self.calibrations, key=lambda k: len(self.calibrations[k])) # HN with most dates
        
        some_certificate = self.get_sertificate( tmp_cone_number, today )

        some_certificate['cone_number'] = cone_number # replace cone number
        some_certificate['alfa'] = 1
        some_certificate['beta'] = 0
        return some_certificate



    def get_sertificate( self, cone_number, date ):
        cone_number = int( cone_number ) # keys are ints
        if cone_number in self.calibrations:
            certificates = self.calibrations[cone_number]
            active_certificate = self.calc_active_certificate( list(certificates.keys()), date )
            if active_certificate:
                certificates[active_certificate]['cone_number'] = cone_number
                return certificates[active_certificate]
            raise Exception( 'Sorry, provided date (' + str(date) + ') is before first calibration.' )
        raise Exception( 'No calibration data for cone ' + str(cone_number) + '.' )


    def calc_active_certificate( self, dates, date):
        date_format = '%d-%m-%Y'
        
        date = self.prep_date( date )
        parsed_date = datetime.strptime(date, date_format).date()

        parsed_dates = [ datetime.strptime(some_date, date_format).date() for some_date in dates ]
        parsed_dates.sort() # certificates in ascending order
        
        if parsed_date < parsed_dates[0]: # can't be before first
            return None
        
        for i in range(len(parsed_dates)-1):
            if parsed_date < parsed_dates[i+1]:
                return parsed_dates[i].strftime(date_format)

        return parsed_dates[-1].strftime(date_format) # last one should be active


    def prep_date( self, date): # desired date format: DD-MM-YYYY
        # replacements
        date = date.replace('/', '-') # needed
        date = date.replace('.', '-') # needed
        date = date.replace('\\', '-')        
        date = date.replace(',', '-')

        # add possible fix for formats? YYYY-MM-DD, MM-DD-YYYY, DD-MM-YY (has to be obvious)
        # dates controlled in this project, so not needed here.
        return date


if __name__=='__main__':
    cals = calibration_data()

    # validations
    #print( cals.get_sertificate( 4458, '17-11-2011') ) # error:  before first calibration -> OK
    #print( cals.get_sertificate( 4458, '32-05-2017') ) # error:  date format not recognized -> OK

    print( cals.get_sertificate( 4458, '18-11-2011') ) # 18-11-2011 -> OK
    print( cals.get_sertificate( 4458, '31-05-2017') ) # 09-05-2016 -> OK    
    print( cals.get_sertificate( 4458, '02-06-2017') ) # 01-06-2016 -> OK