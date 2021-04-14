'''
Utility to generate python argparse based on a given dictionary
'''

import argparse, sys
import special_printer as sp

def str2bool(v):
  '''
  To handle boolean arguments. type=bool in argparse does not work
  '''
  result = (v.lower() in ("yes", "true", "t", "1"))
  return result

def generate_then_parse_arguments(info, skip=[]):
    '''
    Supported argument types: string, int, float, 1D list and 1D tuple
    '''

    ### Generate argparse ####
    parser = argparse.ArgumentParser(description='Auto-generated argparse content')

    for i in info:
        if i in skip:
            continue

        if isinstance(info[i], str):
            parser.add_argument('--' + i, type=str)
        elif isinstance(info[i], bool):
            print(i)
            parser.add_argument('--' + i, type=str2bool)
        elif isinstance(info[i], int):
            parser.add_argument('--' + i, type=int)
        elif isinstance(info[i], float):
            parser.add_argument('--' + i, type=float)

        elif isinstance(info[i], tuple) or isinstance(info[i], list):
            parser.add_argument('--' + i, nargs='+')

    args = parser.parse_args()

    ### Assign command line arguments back ###
    global has_arg
    for i in info:
        if i in skip:
            continue

        has_arg = None
        if isinstance(info[i], str) or isinstance(info[i], int) or \
            isinstance(info[i], float) or isinstance(info[i], bool):
            exec('global has_arg; has_arg = args.' + i)
            if has_arg is not None:
                exec('info[i] = args.' + i)
        elif isinstance(info[i], tuple):
            exec('global has_arg; has_arg = args.' + i)
            if has_arg is not None:
                exec('info[i] = args.' + i)
                info[i] = tuple(info[i])
        elif isinstance(info[i], list):
            exec('global has_arg; has_arg = args.' + i)
            if has_arg is not None:
                exec('info[i] = args.' + i)
                info[i] = list(info[i])

    #sp.info(str(info))
    return info



#### Testers #####


if __name__ == '__main__':
    import pprint as pp

    info = {
            'string_arg': 'arg1',\
            'int_arg': 2,\
            'float_arg': 3.0,\
            'tuple_arg': (4,0,1),\
            'list_arg': [5,0,1],\
            'not_supported': None,\
            'bool_arg': False
    }

    bad = ['not_supported']
    new_info = generate_then_parse_arguments(info, bad)
    pp.pprint(new_info)
