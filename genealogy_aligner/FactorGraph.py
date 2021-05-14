import igraph as ig
import numpy as np

class factor_graph:
    def __init__(self):
        self._graph = ig.Graph ()

        def add_factor_node(self, f_name, factor_): pass

        def change_factor_distribution(self, f_name, factor_): pass

def add_factor_node(self, f_name, factor_):
    if (self.get_node_status ( f_name ) != False) or (f_name in factor_.get_variables ()):
        raise Exception ( 'Invalid factor name' )
    if type ( factor_ ) is not factor:
        raise Exception ( 'Invalid factor_' )
    for v_name in factor_.get_variables ():
        if self.get_node_status ( v_name ) == 'factor':
            raise Exception ( 'Invalid factor' )

    # Check ranks
    self.__check_variable_ranks ( f_name, factor_, 1 )
    # Create variables
    for v_name in factor_.get_variables ():
        if self.get_node_status ( v_name ) == False:
            self.__create_variable_node ( v_name )
    # Set ranks
    self.__set_variable_ranks ( f_name, factor_ )
    # Add node and corresponding edges
    self.__create_factor_node ( f_name, factor_ )

factor_graph.add_factor_node = add_factor_node

def change_factor_distribution(self, f_name, factor_):
    if self.get_node_status ( f_name ) != 'factor':
        raise Exception ( 'Invalid variable name' )
    if set ( factor_.get_variables () ) != set ( self._graph.vs[self._graph.neighbors ( f_name )]['name'] ):
        raise Exception ( 'invalid factor distribution' )

    self.__check_variable_ranks ( f_name, factor_, 0 )
    self.__set_variable_ranks ( f_name, factor_ )
    self._graph.vs.find ( name=f_name )['factor_'] = factor_

factor_graph.change_factor_distribution = change_factor_distribution


def string2factor_graph(str_):
    res_factor_graph = factor_graph ()

    str_ = [i.split ( '(' ) for i in str_.split ( ')' ) if i != '']
    for i in range ( len ( str_ ) ):
        str_[i][1] = str_[i][1].split ( ',' )

    for i in str_:
        res_factor_graph.add_factor_node ( i[0], factor ( i[1] ) )

    return res_factor_graph



