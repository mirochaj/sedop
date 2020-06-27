"""
ReadParameterFile.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2011-01-28.

Description: Read in parameter file.  Stole from rt1d, no ProblemType option though.
     
"""

import re
from .SetDefaultParameterValues import *  

def ReadParameterFile(pf):
    """
    Read in the parameter file, and parse the parameter names and arguments.
    Return a dictionary that contains all parameters and their values, whether 
    they be floats, tuples, or lists.
    """
    f = open(pf, "r")
    pf_dict = SetDefaultParameterValues()
    for line in f:
        if not line.split(): 
            continue
        if line.split()[0][0] == "#": 
            continue
        
        # This will prevent crashes if there is not a blank line at the end of the parameter file
        if line[-1] != '\n': 
            line += '\n'
        
        # Cleave off end-of-line comments.
        line = line[:line.rfind("#")].strip()
        
        # Read in the parameter name and the parameter value(s).
        parname, eq, parval = line.partition("=")
                                
        # Else, actually read in the parameter                                     
        try: 
            parval = float(parval)
        except ValueError:
            if re.search('/', parval):  # For directory with more than one level
                parval = str(parval.strip())
            elif parval.strip().isalnum(): 
                parval = str(parval.strip())
            elif parval.replace('_', '').strip().isalnum():
                parval = parval.strip()
            elif parval.partition('.')[-1] in ['dat', 'hdf5', 'h5', 'txt']:
                parval = str(parval.strip())
            else:
                parval = parval.strip().split(",")
                tmp = []       
                if parval[0][0] == '(':
                    for element in parval: 
                        if element.strip(" (,)").isdigit(): 
                            tmp.append(float(element.strip("(,)")))
                        else: 
                            tmp.append(element.strip(" (,)"))
                    parval = tuple(tmp)                    
                elif parval[0][0] == '[':
                    for element in parval: 
                        tmp.append(float(element.strip("[,]")))
                    parval = list(tmp)
                else:
                    print(parname, parval)
                    raise ValueError('The format of this parameter is not understood.')
                
        pf_dict[parname.strip()] = parval
                
    return pf_dict