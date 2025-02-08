from class_tot import tot
import classification as clf
import get_sgf
import os

def plot_tot():
    soundings = [ f for f in os.listdir() if os.path.isfile( f ) ]
    tots = [ s for s in soundings if '.tot' in s.lower() ]
    for t in tots:
            temp_tot_data = get_sgf.read_tot( t , sounding_nr=0 )
            tmp_name = t.lower().replace('.tot', '.png')

            some_tot = tot( 
                sgf_data=temp_tot_data,
                comment='',
                apply_corrections=False,
                calculate_additional_variables=True,
            )

            some_tot.add_classifications( 
                clf.TOT_methods.get_classifications( some_tot.get_tot_data(), name=some_tot.get_name(), folder='img' )
                )

            some_tot.to_figure( tmp_name, color_figure=True, show_classifications = True )

if __name__ == '__main__':
    plot_tot()