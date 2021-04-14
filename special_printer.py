import platform, time, sys, re, datetime, os

# A switch so that logs don't record ANSI escape sequences
logging=False

# Detect system that does not support ANSI
def do_ansi():
    if platform.system() == 'Windows' or logging:
        return False
    else:
        return True


# ANSI colors
BLACK=30
RED=31
GREEN=32
YELLOW=33
BLUE=34
MAGENTA=35
CYAN=36
WHITE=37

# ANSI Font Effects
NORMAL=0
NORMAL_TEXT=22
BOLD=1
FAINT=2
UNDERLINE=4
UNDERSCORE=4
REVERSE=7

# CLEAN
NO_EFFECT='\033[0m'


cur_effects = []
def NC():
    global cur_effects
    cur_effects = []

    if not do_ansi():
        return ''
    else:
        return NO_EFFECT


def effect(effects):
    '''
    a list of effects you want to have. This function constructs
    an ANSI escape sequence
    '''
    global cur_effects

    if not do_ansi():
        return ''
    else:
        wanted_effects = set(effects)
        wanted_effects.update(cur_effects)
        effects = [str(e) for e in wanted_effects]
        starter = '\033['
        end = 'm'
        delim = ';'
        string = starter + delim.join(effects) + end
        cur_effects = effects
        return string


'''
def color(string, c):
    if platform.system() == 'Windows' or logging:
        return string
    else:
        result = '%s%s%s' % (c, string, NC)
        return result
'''

### Logging ###
# Only one log supported at a time. If multiple files import special_printer,
# the last file that defines log_file is the only file that actually defined it
log_file = None
def start_log_file(filename, do_append=False):
    global log_file
    if not do_append:
        log_file = open(filename, 'w')
    else:
        log_file = open(filename, 'a')

    # Since the script might be executed on remote clients, we record UTC time
    log_file.write('Log start time (UTC): ' + str(datetime.datetime.utcnow()) + '\n\n')


def write_log(string, newline=True):
    global logging
    logging = True

    log_file.write(str(string))
    if newline:
        log_file.write('\n')
    log_file.flush()

    logging = False


def logged_print(line):
    sys.stdout.write(line + '\n')
    sys.stdout.flush()

    # Log file should not store any ANSI stuff
    if log_file is not None:
        ansi_escape = re.compile(r'\x1b[^m]*m')
        line=ansi_escape.sub('', line)
        write_log(line)


def warning(message, indent=0, prefix=''):
    line = '  ' * indent + prefix + effect([BOLD, RED]) + 'WARNNING: ' + NC() + message
    logged_print(line)


def info(message, indent=0, prefix=''):
    line = '  ' * indent + prefix + effect([BOLD, BLUE]) + 'INFO: ' + NC() + message
    logged_print(line)


def news(message, indent=0, prefix=''):
    line = '  ' * indent + effect([UNDERLINE]) + prefix + effect([BOLD, YELLOW]) + 'NEWS: ' + effect([NORMAL_TEXT]) + message + NC()
    logged_print(line)


def var_print(name, value, sig_dig=-1):
    if sig_dig != -1:
        value = round(value, sig_dig)
    return effect([YELLOW, REVERSE]) + name + '=' + str(value) + NC() + ' '


def heading(message, indent=0, line='-', do_print=True):
    line = '  ' * indent + effect([BOLD]) + line * 4 + NC() + ' ' + \
                    effect([BOLD, GREEN]) + message + NC() + ' ' + \
                    effect([BOLD]) + line * 4 + NC()
    if do_print:
        print(line)
        sys.stdout.flush()
    else:
        return line


def color_string(string, color_string):
    c = eval(color_string)
    return color(string, c)



def flush_print(f, string, new_line=False):
    if new_line == False:
        f.write(string)
    else:
        f.write(string + '\n')
    f.flush()


#### !!!WARNING!!! Haven't fixed progreee related prints for ANSI coloring yet ###

pid = -1
pheader = ''
prev_len = 0
def progress(decimal, header, more, indent=0, prefix='', cid=0):
    global pid
    global pheader
    global prev_len

    pheader = header
    pid = cid
    percent = round(decimal * 100, 2)

    message = '  ' * indent + prefix + color(header + ': ', LIGHTBLUE) + \
                progress_bar(percent) + str(percent) + '% ' + more
    pad_length = max(0, prev_len - len(message))
    message = message + ' ' * pad_length
    prev_len = len(message)
    if cid != pid:
        print(message)
    else:
        print(message, end='\r', flush=True)


def progress_bar(percent, length=20, done='=', prog='>', fut='#'):
    done_count = int(round(length * percent / 100)) - 1
    fut_count = max(0, length - 1 - done_count)
    result = '[' + done * done_count + prog + fut * fut_count  + '] '
    return result


def progress_done(detail=0):
    global pid
    global pheader
    global prev_len

    if detail == 0:
        print('\n' + info(color(pheader, LIGHTBLUE) + ' is done'))
    elif detail == 1:
        print('')
    else:
        print(' ' * prev_len, end='\r', flush=True)
    pheader = ''
    pid = -1
    prev_len = 0


def progress_debug(message):
    fill_length = max(0, prev_len - len(message))
    print(message + ' ' * fill_length, flush=True)
    sys.stdout.flush()





################## Testers ##################
def test_progress():
    for i in range(0, 11):
        progress(i * 0.1, 'test', cid=1)
        time.sleep(0.25)
    progress_done(0)
    for i in range(0, 101):
        progress(i * 0.01, 'another test', cid=1)
        time.sleep(0.02)
    progress_done()


def standard_test():

    start_log_file('test_sp.log')

    warning('Test Warning')
    info('Test info')
    news('Test News', prefix='Interesting Stuff ')
    info(var_print('test_var', 1.23456789, 3))
    news(var_print('test_var', 1.23456789, 3))

if __name__ == '__main__':
    standard_test()
