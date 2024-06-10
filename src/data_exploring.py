import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../lib')))

from lib_import import plt, sns, FuncFormatter

def thousands_formatter(x, pos):
    return f'{x*1e-3:.2f}k'