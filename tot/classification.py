import math
color_palette = { # defined as (R, G, B) and edited to % for matplotlib below
    'outside': ( 255,255,255 ),

# by fractions ( adapted from SVV CPTu spreadsheet )
    'organic': ( 100,  50, 0 ),  'clay':    ( 255, 255, 0 ),
    'silt':    ( 255, 127, 0 ),  'sand':    (   0, 255, 0 ),
    'gravel':  (   0, 127, 0 ),

# by sensitivity
    'quick':         (   0, 255, 255 ),
    'brittle':       ( 112,  48, 160 ),
    'not sensitive': (   0, 255,   0 ), # green (from sand)

# by hardness
    'very loose':  ( 100,  50, 0 ), # as organic
    'loose':       ( 255, 255, 0 ), # as clay
    'medium hard': ( 255, 127, 0 ), # as silt
    'hard':        (   0, 255, 0 ), # as sand
    'very hard':   (   0, 127, 0 ), # as gravel
}


# calculate % values for colors for plotting in matplotlib
for color in color_palette:
    tmp_color = list( color_palette[color] )
    tmp_color = [x/255 for x in tmp_color]
    color_palette[color] = tuple(tmp_color)


tot_1D_meths = [{ # binary params. (e.g. hammering)- 0:Off / 1:Optional / 2:On
        'name': 'NGI 2019',
        'ref': 'Quinteros et al. (2019) - Øysand research site: Geotechnical characterisation of deltaic sandy-silty soils',
        'var': 'A',
        'regions':{
                1: { 'name':'Clay/Silt', 'min': 0, 'max': 5, 'incr_rpm': 0, 'flush': 0, 'hammer': 0, 'color': color_palette['clay'] },
                2: { 'name':'Silt/Sand', 'min': 5, 'max': 10, 'incr_rpm': 0, 'flush': 0, 'hammer': 0, 'color': color_palette['silt'] },
                3: { 'name':'Sand', 'min': 10, 'max': 20, 'incr_rpm': 0, 'flush': 0, 'hammer': 0, 'color': color_palette['sand'] },
                4: { 'name':'Sand/Gravel', 'min': 20, 'max': 30, 'incr_rpm': 0, 'flush': 0, 'hammer': 0, 'color': color_palette['gravel'] }
                }
        },{
        'name': 'Norconsult 2015',
        'ref': 'Klassifisering av grunnens fasthet ut fra boremotstand med diverse metoder',
        'var': 'A',
        'regions':{ # not entirely successful implementation as class #6 requires less than 3m/min penetration speed
                1: { 'name':'Meget løst/bløtt', 'min': 0, 'max': 2, 'incr_rpm': 0, 'flush': 0, 'hammer': 0, 'color': color_palette['very loose'] },
                2: { 'name':'Løst/bløtt', 'min': 2, 'max': 10, 'incr_rpm': 0, 'flush': 0, 'hammer': 0, 'color': color_palette['loose'] },
                3: { 'name':'Middels fast', 'min': 10, 'max': 25, 'incr_rpm': 0, 'flush': 0, 'hammer': 0, 'color': color_palette['medium hard'] },
                4: { 'name':'Fast', 'min': 25, 'max': 999, 'incr_rpm': 0, 'flush': 0, 'hammer': 0, 'color': color_palette['hard'] },
                5: { 'name':'Fast', 'min': 0, 'max': 25, 'incr_rpm': 2, 'flush': 1, 'hammer': 1, 'color': color_palette['hard'] },
                6: { 'name':'Meget fast', 'min': 10, 'max': 999, 'incr_rpm': 2, 'flush': 2, 'hammer': 2, 'color': color_palette['very hard'] }
                }
        }]


tot_2D_meths = [
        { # binary params. (e.g. hammering)- 0:Off / 1:Optional / 2:On
            'name': 'SVV 2016',
            'ref': 'Haugen et al (2016) - A preliminary attempt towards soil classification chart from total sounding',
            'var_x': 'q_ns-0_3',
            'var_y': 'std_fdt-0_3',
            'log_x': True,
            'log_y': True,
            'regions':{
                1: {
                    'name':'Quick clay', 
                    'x':[ 0.1, 2, 12, 30, 50, 0.1, 0.1 ],
                    'y': [ 0.09, 0.09, 0.09, 0.01, 0.001, 0.001, 0.09 ],
                    'incr_rpm': 0, 'flush': 0, 'hammer': 0,
                    'color': color_palette['quick']
                    },
                2: {
                    'name':'Clay',
                    'x':[ 0.1, 4, 30, 80, 200, 50, 30, 12, 2, 0.1, 0.1 ],
                    'y': [ 3, 1.5, 0.3, 0.05, 0.001, 0.001, 0.01, 0.09, 0.09, 0.09, 3 ],
                    'incr_rpm': 0, 'flush': 0, 'hammer': 0,
                    'color': color_palette['clay'] 
                    },
                3: {
                    'name':'Silt',
                    'x':[ 0.1, 10, 100, 300, 1000, 200, 80, 30, 4, 0.1, 0.1 ],
                    'y': [ 100, 12, 1, 0.1, 0.001, 0.001, 0.05, 0.3, 1.5, 3, 100 ],
                    'incr_rpm': 0, 'flush': 0, 'hammer': 0,
                    'color': color_palette['silt']
                    },
                4: {
                    'name':'Sand',
                    'x':[ 0.1, 100, 3000, 10000, 10000, 1000, 300, 100, 10, 0.1, 0.1 ],
                    'y': [ 1000, 1000, 30, 1, 0.001, 0.001, 0.1, 1, 12, 100, 1000 ],
                    'incr_rpm': 0, 'flush': 0, 'hammer': 0,
                    'color': color_palette['sand']
                    }
                }
        }]


class _methods():
    def __init__( self, method_type='tot' ):
        self.methods = []
        self.method_type = method_type

    def add_methods( self, methods, dim):
        for method in methods:
            if self.method_type == 'tot':
                if dim == 1:
                    self.methods.append( _1d_tot_method(method) )
                elif dim ==2:
                    self.methods.append( _2d_tot_method(method) )
            else:
                pass

    def no_methods( self ):
        return len( self.methods )

    def get_method( self, index ):
        return self.methods[ index ]
    
    def get_classifications( self, method_data, name=None, folder=None ):
        res = []

        for method in self.methods:            
            res.append( method.get_classification( method_data, name=name, folder=folder ))
        
        return res


class _2d_tot_method():
    def __init__( self, definition ):
        self.name = definition['name']
        self.ref = definition['ref']
        self.var_x = definition['var_x']
        self.var_y = definition['var_y']
        self.log_x = definition['log_x']
        self.log_y = definition['log_y']

        self.regions = self.add_regions( definition )
    
    def add_regions( self, method_data ):
        regions = []
        region_data = method_data['regions']

        for r in region_data:
            regions.append( _2d_region(
                nr = r,
                name = region_data[r]['name'],
                x = region_data[r]['x'],
                y = region_data[r]['y'],
                log_x = method_data['log_x'],
                log_y = method_data['log_y'],
                incr_rpm = region_data[r]['incr_rpm'],
                flush = region_data[r]['flush'],
                hammer = region_data[r]['hammer'],
                color=region_data[r]['color']
            ))
        return regions

    def classify( self, tot_data ):
        X = tot_data[self.var_x].tolist()
        Y = tot_data[self.var_y].tolist()

        B  = tot_data['B_org'].tolist() # Unaltered rate of penetration (mm/s)
        AP = tot_data['AP'].tolist() # Hammering
        AR = tot_data['AR'].tolist() # Flushing
        AQ = tot_data['AQ'].tolist() # Increased rotation

        res = []

        for some_x, some_y, rate, incr_rpm,flush,hammer in zip(X, Y, B, AQ, AR, AP):
            classified = False
            for region in self.regions:
                if region.contains( some_x, some_y, rate, incr_rpm, flush, hammer ):
                    res.append( region.nr() )
                    classified = True
                    break

            if not classified:
                res.append( -1 )
        return res

    def get_short_profile( self, tot_data ):
        depth = tot_data['D'].tolist()

        long_profile = self.classify( tot_data )

        short_D = []
        short_profile = []

        for d, pr in zip( depth, long_profile ):
            if not short_D: # list empty
                short_D.append(d)
                short_profile.append(pr)
            elif pr != short_profile[-1]: # new classification
                short_D.append(d)
                short_profile.append(pr)
            # most points ignored ( do not satisfy if/elif )

        # add last depth if not new classification
        if short_D[-1] != depth[-1]: 
            short_D.append( depth[-1] )
            short_profile.append( short_profile[-1] )

        # more correct to add final increment - drop this for now
        return short_D, short_profile
    
    def get_legend( self ):
        res = {}
        for region in self.regions:
            res[region.nr()] = { 
                'name': region.name(),
                'color': region.color()
                }
        # add region/color for points outside model
        res[-1] = {
            'name': '',
            'color': color_palette['outside']
        }
        return res
    
    def get_classification( self, tot_data, name=None, folder=None ):
        d, profile = self.get_short_profile( tot_data )
        legend = self.get_legend()
        
        res = {
            'name': self.name,
            'results': {
                'depth': d,
                'class': profile
            },
            'legend': legend
        }

        if name: # plot model
            self.save_model_figure( tot_data, name, folder )
            
        return res
    
    def save_model_figure( self, tot_data, name, folder ):
        import os
        import matplotlib.pyplot as plt

        filename = os.path.join( folder,"model_" + str(name) + ".png" )

        x_vals = tot_data[self.var_x].tolist()
        y_vals = tot_data[self.var_y].tolist()
        fig_name = self.name

        # create/define figure
        fig, ax = plt.subplots()
        if self.log_x:
            ax.set_xscale('log')
        if self.log_y:
            ax.set_yscale('log')

        # annotate
        fig.suptitle( str(name) + " (" + fig_name + ")")
        ax.set_xlabel(self.var_x)
        ax.set_ylabel(self.var_y)

        # draw model regions
        for r in self.regions:
            region_x, region_y = r.get_coords()
            ax.plot( region_x, region_y, color=r.color() )

        # draw model points
        ax.scatter( x_vals, y_vals, marker='o', s=2, c='tab:brown')
        
        # save figure
        plt.savefig( filename,dpi=300, transparent=True )
        plt.close()


class _1d_tot_method():
    def __init__( self, definition ):
        self.name = definition['name']
        self.ref = definition['ref']

        self.regions = self.add_regions( definition['regions'] )
    
    def add_regions( self, region_data ):
        regions = []

        for r in region_data:
            regions.append( _1d_region(
                nr = r,
                name = region_data[r]['name'],
                min_f = region_data[r]['min'],
                max_f = region_data[r]['max'],
                incr_rpm = region_data[r]['incr_rpm'],
                flush = region_data[r]['flush'],
                hammer = region_data[r]['hammer'],
                color=region_data[r]['color']
            ))
        return regions

    def classify( self, tot_data ):
        A  = tot_data['A'].tolist() # Push force
        B  = tot_data['B_org'].tolist() # Unaltered rate of penetration (mm/s)
        AP = tot_data['AP'].tolist() # Hammering
        AR = tot_data['AR'].tolist() # Flushing
        AQ = tot_data['AQ'].tolist() # Increased rotation

        res = []

        for f_dt, rate, incr_rpm,flush,hammer in zip(A,B, AQ, AR, AP):
            classified = False
            for region in self.regions:
                if region.contains( f_dt, rate, incr_rpm, flush, hammer ):
                    res.append( region.nr() )
                    classified = True
                    break

            if not classified:
                res.append( -1 )
        return res

    def get_legend( self ):
        res = {}
        for region in self.regions:
            res[region.nr()] = { 
                'name': region.name(),
                'color': region.color()
                }
        # add region/color for points outside model
        res[-1] = {
            'name': '',
            'color': color_palette['outside']
        }
        return res


    def get_short_profile( self, tot_data ):
        depth = tot_data['D'].tolist()

        long_profile = self.classify( tot_data )

        short_D = []
        short_profile = []

        for d, pr in zip( depth, long_profile ):
            if not short_D: # list empty
                short_D.append(d)
                short_profile.append(pr)
            elif pr != short_profile[-1]: # new classification
                short_D.append(d)
                short_profile.append(pr)
            # most points ignored ( do not satisfy if/elif )

        # add last depth if not new classification
        if short_D[-1] != depth[-1]: 
            short_D.append( depth[-1] )
            short_profile.append( short_profile[-1] )

        # more correct to add final increment - drop this for now
        return short_D, short_profile
    
    def get_classification( self, tot_data, name=None, folder=None ): # name&folder unused
        d, profile = self.get_short_profile( tot_data )
        legend = self.get_legend()
        
        res = {
            'name': self.name,
            'results': {
                'depth': d,
                'class': profile
            },
            'legend': legend
        }
        return res


class _2d_region():
    def __init__( self, name, nr, x, y, log_x, log_y, incr_rpm, flush, hammer, color ):
        self.region_name = name
        self.region_nr = nr
        self.x = x
        self.y = y
        self.log_x = log_x
        self.log_y = log_y
        self.incr_rpm =incr_rpm
        self.flush = flush
        self.hammer = hammer
        self.region_color = color

        self.max_x = max(self.x)
        self.min_x = min(self.x)
        self.max_y = max(self.y)
        self.min_y = min(self.y)

    def nr( self ):
        return self.region_nr

    def name( self ):
        return self.region_name

    def color( self ):
        return self.region_color
    
    def get_coords( self ):
        return self.x, self.y

    def xOfyByPoints(self, y, x1, y1, x2, y2):
        if ( self.log_y and y1>0 and y2>0 ):
            if ( self.log_x and y>0 and x1>0 and x2>0 ):
                # log-log
                x = 10 ** ( math.log10( y/y1 ) * math.log10( x1/x2 ) / math.log10( y1/y2 ) + math.log10( x1 ) )
            else: 
                # lin-log
                x = math.log10( y/y1 ) * ( (x1 - x2) / math.log10(y1/y2) ) + x1
        else:
            if ( self.log_x and y>0 and x1>0 and x2>0 ):
                # log-lin
                x = 10 ** ( (y - y1) * math.log10(x1/x2) / (y1 - y2) + math.log10(x1) )
            else:
                # lin-lin ( log(n) where n<=0 will also default here )
                x = (y-y1) * ( (x1-x2) / (y1-y2) ) + x1
        return x

    def contains( self, some_x, some_y, rate, incr_rpm, flush, hammer ):
# ****** not implemented ******
        _ = rate
        _ = incr_rpm
        _ = flush
        _ = hammer
# *****************************
        contains = False
        
        n = len(self.x)
        j = n-1

        if ( some_x < self.min_x or some_x > self.max_x or some_y < self.min_y or some_y > self.max_y ):
            pass # it's outside
        else: # point in poly routine
            for i in range(n):
                if ( ((self.y[i] > some_y) != (self.y[j] > some_y)) and (some_x < self.xOfyByPoints(some_y, self.x[i], self.y[i], self.x[j], self.y[j])) ):
                    contains = not contains
                j=i
        
        return contains


class _1d_region():
    def __init__( self, name, nr, min_f, max_f, incr_rpm, flush, hammer, color ):
        self.region_name = name
        self.region_nr = nr
        self.min_f = min_f
        self.max_f = max_f
        self.incr_rpm =incr_rpm
        self.flush = flush
        self.hammer = hammer
        self.region_color = color

    def nr( self ):
        return self.region_nr

    def name( self ):
        return self.region_name

    def color( self ):
        return self.region_color

    def contains( self, f_dt, rate, incr_rpm, flush, hammer ):
        if (self.incr_rpm*incr_rpm > 0) and (self.flush*flush > 0) and (self.hammer*hammer > 0): # everything on
            standard_rate = 50 # mm/s
            if ( rate <= 0.8 * standard_rate ) and (f_dt >= self.min_f) : # measured rate! (unaltered)
                return True
        elif (self.incr_rpm*incr_rpm > 0) and (f_dt < self.max_f):
            return True
        else:
            return ( (f_dt >= self.min_f) and (f_dt < self.max_f) )


TOT_methods = _methods( method_type='tot' )
TOT_methods.add_methods( methods=tot_1D_meths, dim=1 )
TOT_methods.add_methods( methods=tot_2D_meths, dim=2 )