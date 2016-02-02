"""
Neuronal Dynamics Library

Contains:
MorrisLecar Generator (Clewley PyDSTool lib)
MorrisLecar Type I parameters
MorrisLecar Type II parameters

HodgkinHuxley Generator (Clewley PyDSTool lib)
HodgkinHuxley Type I parameters
HodgkinHuxley Type II parameters

Butera1999 Generator

Connor 1977 Generato

Integrate and Fire Generator (Clewley PyDSTool lib)
"""

from PyDSTool import *

import numpy as np
import matplotlib.pyplot as plt


plt.close('all')
plt.ion()


# Assimilated from Clewley PyDSTool library
"""
Tsumoto, K., Kitajima, H., Yoshinaga, T., Aihara, K., & Kawakami, H. (2006). Bifurcations in Morris–Lecar neuron model. Neurocomputing, 69(4), 293-316.
"""

def makeMorrisLecarNeuron(intType, pars=None, ics=None, strName='MorrisLecar'):
    auxfndict = {'minf': (['v'], '0.5*(1 + tanh((v-v1)/v2))'), \
                 'winf': (['v'], '0.5*(1 + tanh((v-v3)/v4))'), \
                 'tau': (['v'], '1/cosh((v-v3)/(2*v4))'), \
                 ### This is modified from PyDSTool's forced_spring.py example
                 'I': (['t'], 'if(t>=t_on,Iapp,0)*if(t>t_off,0,1)') \
                   }
    
    vstr = '(Iapp - gCa*minf(v)*(v-vCa) - gK*w*(v-vK) - gL*(v-vL))/C'
    wstr = 'phi*(winf(v)-w)/tau(v)'
    iExtstr = 'I(t)'
    
    DSargs = args(name='MorrisLecar')
    DSargs.varspecs = {'v': vstr, 'w': wstr, 'Iext': iExtstr}
    DSargs.auxvars = ['Iext']
    DSargs.fnspecs = auxfndict
    DSargs.tdata = [0,250]
    
    dsIcdict = None
    dsPars = {'t_on': DSargs.tdata[0],
              't_off': DSargs.tdata[-1]
              }
    
    if intType == 1:
        # "Computational Cell Biology", Fall (Type I)
        dsPars.update({'Iapp': 0.0,
                'C': 20.,
                'vK': -84.,
                'gK': 8.,
                'vCa': 120.,
                'gCa': 4.,
                'vL': -60.,
                'gL': 2.,
                'v1': -1.2,
                'v2': 18.,
                'v3': 12.,
                'v4': 17.4,
                'phi': 0.066})
        
        dsIcdict = {'v': -60., 'w': 0.01}
    elif intType == 2:
        # "Computational Cell Biology", Fall (Type II)
        dsPars.update({'Iapp': 0.0,
                'C': 20.,
                'vK': -84.,
                'gK': 8.,
                'vCa': 120.,
                'gCa': 4.4,
                'vL': -60.,
                'gL': 2.,
                'v1': -1.2,
                'v2': 18.,
                'v3': 2.,
                'v4': 30.,
                'phi': 0.04})
        
        dsIcdict = {'v': -60., 'w': 0.01} 
    else:
        raise ValueError("Invalid excitability Type: Either 1 or 2")
    
    if pars is not None:
        dsPars.update(pars)
    if ics is not None:
        dsIcdict.update(ics)
    
    DSargs.pars = dsPars
    DSargs.ics = dsIcdict    
    
    return Generator.Vode_ODEsystem(DSargs)
  
    
# Persistent Sodium Current
# Butera, Rinzel, Smith (1999). Models of respiratory rhythm generation in the
# Pre-Botzinger Complex. I. Bursting pacemaker neurons. J Neurophysiol 82: 382-
# 397.    
    
def makeButeraNeuron(name, par_args=None, ic_args=None):
    
    DSargs = args()    
    
    vfn_str = "(I(t)-gL*(v-EL)-gNa*minf(v)*minf(v)*minf(v)*(1-n)*(v-ENa)-gK*n*n*n*n*(v-EK)-gNaP*mpinf(v)*hp*(v-ENaP))/Cm"
    nfn_str = '(ninf(v)-n)/ntau(v)'
    hpfn_str = '(hpinf(v)-hp)/hptau(v)'
    ILeak_fn_str = 'gL*(v-EL)'
    INa_fn_str = 'gNa*minf(v)*minf(v)*minf(v)*(1-n)*(v-ENa)'
    IK_fn_str = 'gK*n*n*n*n*(v-EK)'
    INaP_fn_str = 'gNaP*mpinf(v)*hp*(v-ENa)'
    Iext_fn_str = 'I(t)'

    
    aux_dict = {
        ### These equations are from Butera 1999
        
        'minf': (['v'], '1/(1+exp(-(v+34)/5))'),
        
        'ninf': (['v'], '1/(1+exp(-(v+29)/4))'),
        'ntau': (['v'], '10/cosh(-(v+29)/(2*4))'),
        
        'mpinf': (['v'], '1/(1+exp(-(v+40)/6))'),
        
        'hpinf': (['v'], '1/(1+exp((v+48)/6))'),
        'hptau': (['v'], '10000/cosh((v+48)/(2*6))'),
        
        ### This is modified from PyDSTool's forced_spring.py example
        'I': (['t'], 'if(t>=t_on,Iapp,0)*if(t>t_off,0,1)')
    }    
    

    DSargs.varspecs = {
        'v': vfn_str,
        'n': nfn_str,
        'hp': hpfn_str,
        'ILeak': ILeak_fn_str,
        'INa': INa_fn_str,
        'IK': IK_fn_str,
        'INaP': INaP_fn_str,
        'Iext': Iext_fn_str
    }
    DSargs.auxvars = ['ILeak', 'INa', 'IK', 'INaP', 'Iext']
    DSargs.fnspecs = aux_dict
    DSargs.xdomain = {
        'v': [-100,80],
        'n': [0,1],
        'hp': [0,1]
    }
    
    if par_args == None:
        # Pars as they appear in Connor 1977 
        DSargs.pars = {
            'Cm': 21.,
            'gL': 2.8, 'EL': -65,
            'gNa': 28, 'ENa': 50,
            'gK': 11.2, 'EK': -85,
            'gNaP': 2.8, 'ENaP': 50,
            'Iapp': 0., 't_on': 0., 't_off': 100.
        }
    elif par_args != None:
        DSargs.pars = par_args
    
    if (ic_args == None) or (ic_args == 'naive'):
        # ICS for first simulation
        DSargs.ics = {
            'v': -65,
            'n': 0.,
            'hp': 0.9
        }
    elif ic_args == 'ss':
        # ICS at steady-state following one spike from "first simulation" above
        DSargs.ics = {
            'v': -53.3206513822,
            'n': 0.00228211136318,
            'hp': 0.659694407505
        }        
    elif ic_args != None:
        DSargs.ics = ic_args
        
    DSargs.name = name
    DSargs.tdata = [0,100]
    
    return Generator.Vode_ODEsystem(DSargs)

# Persistent Sodium Current & Non-Inactivating Potassium Current
#
# Persistent Sodium Current
# Butera, Rinzel, Smith (1999). Models of respiratory rhythm generation in the
# Pre-Botzinger Complex. I. Bursting pacemaker neurons. J Neurophysiol 82: 382-
# 397.  
#
# Non-Inactivating Potassium Current
# Hill, Lu, Masino, Olsen, Calabrese (2001). A model of a segmental oscillator
# in the leech heartbeat neuronal network. J Comp Neurosci 10: 281-302.

def makeModifiedButeraNeuron(name, par_args=None, ic_args=None):
    
    DSargs = args()    
    
    vfn_str = "(I(t)-gL*(v-EL)-gNa*minf(v)*minf(v)*minf(v)*(1-n)*(v-ENa)-gK*n*n*n*n*(v-EK)-gNaP*mpinf(v)*hp*(v-ENaP)-gK2*mk2*mk2*(v-EK))/Cm"
    nfn_str = '(ninf(v)-n)/ntau(v)'
    hpfn_str = '(hpinf(v)-hp)/hptau(v)'
    mk2fn_str = '(mk2inf(v)-mk2)/mk2tau(v)'
    ILeak_fn_str = 'gL*(v-EL)'
    INa_fn_str = 'gNa*minf(v)*minf(v)*minf(v)*(1-n)*(v-ENa)'
    IK_fn_str = 'gK*n*n*n*n*(v-EK)'
    INaP_fn_str = 'gNaP*mpinf(v)*hp*(v-ENa)'
    IK2_fn_str = 'gK2*mk2*mk2*(v-EK)'
    Iext_fn_str = 'I(t)'

    
    aux_dict = {
        ### These equations are from Butera 1999
        
        'minf': (['v'], '1/(1+exp(-(v+34)/5))'),
        
        'ninf': (['v'], '1/(1+exp(-(v+29)/4))'),
        'ntau': (['v'], '10/cosh(-(v+29)/(2*4))'),
        
        'mpinf': (['v'], '1/(1+exp(-(v+40)/6))'),
        
        'hpinf': (['v'], '1/(1+exp((v+48)/6))'),
        'hptau': (['v'], '10000/cosh((v+48)/(2*6))'),
        
        'mk2inf': (['v'], '1/(1+exp(-0.083*(v+20)))'),
        'mk2tau': (['v'], '0.057+0.043/(1+exp(0.2*(v+35)))'),
        
        ### This is modified from PyDSTool's forced_spring.py example
        'I': (['t'], 'if(t>=t_on,Iapp,0)*if(t>t_off,0,1)')
    }    
    

    DSargs.varspecs = {
        'v': vfn_str,
        'n': nfn_str,
        'hp': hpfn_str,
        'mk2': mk2fn_str,
        'ILeak': ILeak_fn_str,
        'INa': INa_fn_str,
        'IK': IK_fn_str,
        'INaP': INaP_fn_str,
        'IK2': IK2_fn_str,
        'Iext': Iext_fn_str
    }
    DSargs.auxvars = ['ILeak', 'INa', 'IK', 'INaP', 'IK2', 'Iext']
    DSargs.fnspecs = aux_dict
    DSargs.xdomain = {
        'v': [-100,80],
        'n': [0,1],
        'hp': [0,1],
        'mk2': [0,1]
    }
    
    
    # Pars as they appear in Connor 1977 
    DSargs.pars = {
        'Cm': 21.,
        'gL': 2.8, 'EL': -65,
        'gNa': 28, 'ENa': 50,
        'gK': 11.2, 'gK2': 50.0, 'EK': -85,
        'gNaP': 2.8, 'ENaP': 50,
        'Iapp': 0., 't_on': 0., 't_off': 100.
    }
    if par_args != None:
        DSargs.pars = DSargs.pars.update(par_args)
    
    if (ic_args == None) or (ic_args == 'naive'):
        # ICS for first simulation
        DSargs.ics = {
            'v': -65,
            'n': 0.,
            'hp': 0.9,
            'mk2': 0.,
        }
    elif ic_args == 'ss':
        # ICS at steady-state following one spike from "first simulation" above
        DSargs.ics = {
            'v': -53.3206513822,
            'n': 0.00228211136318,
            'hp': 0.659694407505,
            'mk2': 0.,
        }        
    elif ic_args != None:
        DSargs.ics = ic_args
        
    DSargs.name = name
    DSargs.tdata = [0,100]
    
    return Generator.Vode_ODEsystem(DSargs)


# Curstacean walking leg model axon neuronal model
# Connor, Walter, McKown (1977). Neural repetitive firing: Modifications of the
# Hodgkin-Huxley axon suggested by experimental results from crustacean axons.
# Biophysical Journal 18: 81.

def makeConnorNeuron(name, par_args=None, ic_args=None):
    
    DSargs = args()    
    
    vfn_str = "(I(t)-gL*(v-EL)-gNa*m*m*m*h*(v-ENa)-gK*n*n*n*n*(v-EK)-gA*A*A*A*B*(v-EK))/Cm"
    mfn_str = '(minf(v)-m)/mtau(v)'
    hfn_str = '(hinf(v)-h)/htau(v)'
    nfn_str = '(ninf(v)-n)/ntau(v)'
    afn_str = '(Ainf(v)-A)/Atau(v)'
    bfn_str = '(Binf(v)-B)/Btau(v)'
    ILeak_fn_str = 'gL*(v-EL)'
    INa_fn_str = 'gNa*m*m*m*h*(v-ENa)'
    IK_fn_str = 'gK*n*n*n*n*(v-EK)'
    IA_fn_str = 'gA*A*A*A*B*(v-EK)'
    Iext_fn_str = 'I(t)'
    
    aux_dict = {
        ### These equations are from Connor 1977
        
        # This is how the equation is originally written in Connor 1977
        #'ma': (['v'], '(-0.1*(v+35-5.3))/exp((-(v+35-5.3)/10)-1)'),
        
        # This is how the equation should be written according to Huxley 1952
        'ma': (['v'], '(-0.1*(v+35+MSHFT))/(exp(-(v+35+MSHFT)/10)-1)'),
        'mb': (['v'], '4*exp(-(v+60+MSHFT)/18)'),
        'minf': (['v'], 'ma(v)/(ma(v)+mb(v))'),
        'mtau': (['v'], '1/(3.8*(ma(v)+mb(v)))'),
        
        
        'ha': (['v'], '0.07*exp(-(v+60+HSHFT)/20)'),
        # This is how the equation is originally written in Connor 1977
        #'hb': (['v'], '1/exp(-((v+30+HSHFT)/10)+1)'),
        
        # This is how the equation should be written according to Huxley 1952        
        'hb': (['v'], '1/(exp(-((v+30+HSHFT)/10))+1)'),
        'hinf': (['v'], 'ha(v)/(ha(v)+hb(v))'),
        'htau': (['v'], '1/(3.8*(ha(v)+hb(v)))'),
        
        # This is how the equation is originally writtein in Connor 1977
        #'na': (['v'], '-0.01*(v+50-4.3)/exp(-((v+50-4.3)/10)-1)'),
        
        # This is how the equation should be written according to Huxley 1952
        'na': (['v'], '-0.01*(v+50+NSHFT)/(exp(-((v+50+NSHFT)/10))-1)'),
        'nb': (['v'], '0.125*exp(-(v+60+NSHFT)/80)'),
        'ninf': (['v'], 'na(v)/(na(v)+nb(v))'),
        'ntau': (['v'], '2/(3.8*(na(v)+nb(v)))'),

        
        'Ainf': (['v'], '(0.0761*(exp((v+94.22)/31.84)/(1+exp((v+1.17)/28.93))))**(1/3)'),
        'Atau': (['v'], '0.3632+(1.158/(1+exp((v+55.96)/20.12)))'),
        
        'Binf': (['v'], '1/(1+exp((v+53.3)/14.54))**4'),
        'Btau': (['v'], '1.24+(2.678/(1+exp((v+50)/16.027)))'),  
        
        ### This is modified from PyDSTool's forced_spring.py example
        'I': (['t'], 'if(t>=t_on,Iapp,0)*if(t>t_off,0,1)')
    }    

    DSargs.varspecs = {
        'v': vfn_str,
        'm': mfn_str,
        'h': hfn_str,
        'n': nfn_str,
        'A': afn_str,
        'B': bfn_str,
        'ILeak': ILeak_fn_str,
        'INa': INa_fn_str,
        'IK': IK_fn_str,
        'IA': IA_fn_str,
        'Iext': Iext_fn_str
    }
    DSargs.auxvars = ['ILeak', 'INa', 'IK', 'IA', 'Iext']
    DSargs.fnspecs = aux_dict
    DSargs.xdomain = {
        'v': [-100,80],
        'm': [0,1],
        'h': [0,1],
        'n': [0,1],
        'A': [0,1],
        'B': [0,1],
    }
    
    if par_args == None:
        # Pars as they appear in Connor 1977 
        DSargs.pars = {
            'Cm': 1.,
            'gL': 0.3, 'EL': -17.,
            'gNa': 120., 'ENa': 55., 'MSHFT': -5.3, 'HSHFT': -12,
            'gK': 20., 'EK': -72., 'NSHFT': -4.3,
            'gA': 47.7,
            'Iapp': 0, 't_on': 0., 't_off': 100.
        }
    elif par_args != None:
        DSargs.pars = par_args
    
    if (ic_args == None) or (ic_args == 'naive'):
        # ICS for first simulation
        DSargs.ics = {
            'v': -65,
            'm': 0.,
            'h': 1.,
            'n': 0.,
            'A': 0.5,
            'B': 0.,
        }
    elif ic_args == 'ss':
        # ICS at steady-state following one spike from "first simulation" above
        DSargs.ics = {
            'v': -64.1331914436,
            'm': 0.0165486511899,
            'h': 0.941065641613,
            'n': 0.199729056641,
            'A': 0.560228361823,
            'B': 0.211436104879,
        }        
    elif ic_args != None:
        DSargs.ics = ic_args
        
    DSargs.name = name
    DSargs.tdata = [0,100]
    
    return Generator.Vode_ODEsystem(DSargs)


# BASE: Curstacean walking leg model axon neuronal model
# Connor, Walter, McKown (1977). Neural repetitive firing: Modifications of the
# Hodgkin-Huxley axon suggested by experimental results from crustacean axons.
# Biophysical Journal 18: 81.
#
# MOD: Persistent Sodium Current
# Hill, Lu, Masino, Olsen, Calabrese (2001). A model of a segmental oscillator
# in the leech heartbeat neuronal network. J Comp Neurosci 10: 281-302.

def makeModifiedConnorNeuron(name, par_args=None, ic_args=None):
    
    DSargs = args()    
    
    # Hill 2001 formulation
    #vfn_str = "(I(t)-gL*(v-EL)-gNa*m*m*m*h*(v-ENa)-gK*n*n*n*n*(v-EK)-gA*A*A*A*B*(v-EK)-gNaP*mNaP*(v-ENa)-gK2*mK2*mK2*(v-EK))/Cm"
    # Butera 1999 formulation
    vfn_str = "(I(t)-gL*(v-EL)-gNa*m*m*m*h*(v-ENa)-gK*n*n*n*n*(v-EK)-gA*A*A*A*B*(v-EK)-gNaP*mNaPinf(v)*hNaP*(v-ENa)-gK2*mK2*mK2*(v-EK))/Cm"
    
    mfn_str = '(minf(v)-m)/mtau(v)'
    hfn_str = '(hinf(v)-h)/htau(v)'
    nfn_str = '(ninf(v)-n)/ntau(v)'
    afn_str = '(Ainf(v)-A)/Atau(v)'
    bfn_str = '(Binf(v)-B)/Btau(v)'
    mK2fn_str = '(mK2inf(v)-mK2)/mK2tau(v)'
    ILeak_fn_str = 'gL*(v-EL)'
    INa_fn_str = 'gNa*m*m*m*h*(v-ENa)'
    IK_fn_str = 'gK*n*n*n*n*(v-EK)'
    IA_fn_str = 'gA*A*A*A*B*(v-EK)'
    
    # Hill 2001 formulation
    #mNaPfn_str = '(mNaPinf(v)-mNaP)/mNaPtau(v)'
    #INaP_fn_str = 'gNaP*mNaP*(v-ENa)'
    # Butera 1999 formulation
    hNaPfn_str = '(hNaPinf(v)-hNaP)/hNaPtau(v)'
    INaP_fn_str = 'gNaP*mNaPinf(v)*hNaP*(v-ENa)'
    
    IK2_fn_str = 'gK2*mK2*mK2*(v-EK)'
    Iext_fn_str = 'I(t)'
    
    aux_dict = {
        ### These equations are from Connor 1977
        
        # This is how the equation is originally written in Connor 1977
        #'ma': (['v'], '(-0.1*(v+35-5.3))/exp((-(v+35-5.3)/10)-1)'),
        
        # This is how the equation should be written according to Huxley 1952
        'ma': (['v'], '(-0.1*(v+35+MSHFT))/(exp(-(v+35+MSHFT)/10)-1)'),
        'mb': (['v'], '4*exp(-(v+60+MSHFT)/18)'),
        'minf': (['v'], 'ma(v)/(ma(v)+mb(v))'),
        'mtau': (['v'], '1/(3.8*(ma(v)+mb(v)))'),
        
        
        'ha': (['v'], '0.07*exp(-(v+60+HSHFT)/20)'),
        # This is how the equation is originally written in Connor 1977
        #'hb': (['v'], '1/exp(-((v+30+HSHFT)/10)+1)'),
        
        # This is how the equation should be written according to Huxley 1952        
        'hb': (['v'], '1/(exp(-((v+30+HSHFT)/10))+1)'),
        'hinf': (['v'], 'ha(v)/(ha(v)+hb(v))'),
        'htau': (['v'], '1/(3.8*(ha(v)+hb(v)))'),
        
        # This is how the equation is originally writtein in Connor 1977
        #'na': (['v'], '-0.01*(v+50-4.3)/exp(-((v+50-4.3)/10)-1)'),
        
        # This is how the equation should be written according to Huxley 1952
        'na': (['v'], '-0.01*(v+50+NSHFT)/(exp(-((v+50+NSHFT)/10))-1)'),
        'nb': (['v'], '0.125*exp(-(v+60+NSHFT)/80)'),
        'ninf': (['v'], 'na(v)/(na(v)+nb(v))'),
        'ntau': (['v'], '2/(3.8*(na(v)+nb(v)))'),

        
        'Ainf': (['v'], '(0.0761*(exp((v+94.22)/31.84)/(1+exp((v+1.17)/28.93))))**(1/3)'),
        'Atau': (['v'], '0.3632+(1.158/(1+exp((v+55.96)/20.12)))'),
        
        'Binf': (['v'], '1/(1+exp((v+53.3)/14.54))**4'),
        'Btau': (['v'], '1.24+(2.678/(1+exp((v+50)/16.027)))'),  

        ## Hill 2001 formulation
        #'mNaPinf': (['v'], '1/(1+exp(-0.12*(v+39)))'),
        #'mNaPtau': (['v'], '10+200/(1+exp(0.4*(v+57)))'),
        
        ## Butera 1999 formulation
        'mNaPinf': (['v'], '1/(1+exp(-(v+40)/6))'),
        'hNaPinf': (['v'], '1/(1+exp((v+48)/6))'),
        'hNaPtau': (['v'], '10000/(cosh((v+48)/(2*6)))'), 

        ## Butera 1999 fit to Delfs 1980
        #'mK2inf': (['v'], '1/(1+exp(-(v+40)/6))'),
        #'mK2tau': (['v'], '0.155/cosh((v+27)/22)'),
        
        ## Hill 2001 formulation
        'mK2inf': (['v'], '1/(1+exp(-0.083*(v+20)))'),
        'mK2tau': (['v'], '57+43/(1+exp(0.2*(v+35)))'),
        
        ### This is modified from PyDSTool's forced_spring.py example
        'I': (['t'], 'if(t>=t_on,Iapp,0)*if(t>t_off,0,1)')
    }    

    DSargs.varspecs = {
        'v': vfn_str,
        'm': mfn_str,
        'h': hfn_str,
        'n': nfn_str,
        'A': afn_str,
        'B': bfn_str,
        # Hill 2001 formulation
        #'mNaP': mNaPfn_str,
        
        # Butera 1999 formulation
        'hNaP': hNaPfn_str,
        
        'mK2': mK2fn_str,
        'ILeak': ILeak_fn_str,
        'INa': INa_fn_str,
        'IK': IK_fn_str,
        'IA': IA_fn_str,
        'INaP': INaP_fn_str,
        'IK2': IK2_fn_str,
        'Iext': Iext_fn_str
    }
    DSargs.auxvars = ['ILeak', 'INa', 'IK', 'IA', 'INaP', 'IK2', 'Iext']
    DSargs.fnspecs = aux_dict
    DSargs.xdomain = {
        'v': [-100,80],
        'm': [0,1],
        'h': [0,1],
        'n': [0,1],
        'A': [0,1],
        'B': [0,1],
        # Hill 2001 formulation
        #'mNaP': [0,1],
        # Butera 1999 formulation
        'hNaP': [0,1],
        'mK2': [0,1]
    }
    
    if par_args == None:
        # Pars as they appear in Connor 1977 
        DSargs.pars = {
            'Cm': 1.,
            'gL': 0.3, 'EL': -17., # Connor 1977: EL = -17.
            'gNa': 120., 'ENa': 55., 'MSHFT': -5.3, 'HSHFT': -12,
            'gK': 20., 'EK': -72., 'NSHFT': -4.3,
            'gA': 47.7,
            'gNaP': 3.,
            'gK2': 50.,
            'Iapp': 0, 't_on': 0., 't_off': 100.
        }
    elif par_args != None:
        DSargs.pars = par_args
    
    if (ic_args == None) or (ic_args == 'naive'):
        # ICS for first simulation
        DSargs.ics = {
            'v': -65,
            'm': 0.,
            'h': 1.,
            'n': 0.,
            'A': 0.5,
            'B': 0.,
            # Hill 2001 formulation
            #'mNaP': 0.,
            # Butera 1999 formulation
            'hNaP': 1.,
            'mK2': 1.
        }
    elif ic_args == 'ss':
        # ICS at steady-state following one spike from "first simulation" above
        DSargs.ics = {
            'v': -63.8079523174,
            'm': 0.0172493718917,
            'h': 0.938348215136,
            'n': 0.203681086206,
            'A': 0.561912130242,
            'B': 0.20550716669,
            'mK2': 0.115422549434,
            'hNaP': 0.116173181942
        }        
    elif ic_args != None:
        DSargs.ics = ic_args
        
    DSargs.name = name
    DSargs.tdata = [0,100]
    
    return Generator.Vode_ODEsystem(DSargs)


def calcFrequency(ts, vs, thresh=0):
    ixs = np.where(np.diff( (vs>=0).astype(int) ) > 0)[0]
    if len(ixs) > 1:
        return 1/np.diff(ts[ixs])
    else:
        return np.array([0])
    
    
def frequencyPlot(gen, freqSampPts, modelName='Neuron Model'):
    
    plt.figure()
    plt.title('Firing Frequency - %s' % modelName)
    
    for s in freqSampPts:
        print "\nSampling Iapp=%.3f" % s
        
        gen.set(pars={'Iapp': s})
        traj = gen.compute('traj')
        pts = traj.sample()
        
        f = calcFrequency(pts['t'], pts['v'])
        print "Frequency: "
        print f
        plt.plot(np.ones(len(f))*s, f, 'b.')
    
    
    plt.draw()
    plt.show()
    

## ==============================
## START TEST CODE HERE
## ==============================

print "Building generator(s)..."
#ML_TypeI = makeMorrisLecarNeuron(1, pars={'Iapp': 80.})
#ML_TypeII = makeMorrisLecarNeuron(2)

#frequencyPlot(ML_TypeI, np.arange(35,100,.5), modelName='Morris Lecar, Type I')
#frequencyPlot(ML_TypeII, np.arange(80,120,.5), modelName = 'Morris Lecar, Type II')

#modButera = makeModifiedButeraNeuron('Modified Butera Neuron')
connor = makeModifiedConnorNeuron('Modified Connor Neuron', ic_args='ss')


## ==============================
## ANTHONY
## ONLY CHANGE THINGS HERE
## ==============================

## This changes the time duration of integration
## Should look like: connor.set(tdata=[<start>, <end>])
## If you break it, copy this: connor.set(tdata=[0, 2500])
## run neuronaldynamicslib.py
connor.set(tdata=[0,15000])

## Play with: Iapp, gNa, gK, gA, gNaP, gK2, gL, EL (all of them...)
## Don't touch: t_on or t_off
## gNa => Fast sodium (Hodgkin-Huxley) (default: 120)
## gK => Fast potassium (Hodgkin-Huxley) (default: 20)
## gA => Transient sodium current (default: 3-11)
## gNaP => Persistent sodium current (default: 0.3 - 8)
## gK2 => M-current (default:find a number?)
## gL => Leak current (default: 0.3 - 5)

connor.set(pars = {
    'Iapp': 10., 't_on': 2000, 't_off': 2010,
    'gNa': 120, 'gK': 20,
    'gA': 3., 'gNaP': 0., 'gK2': 0.,
    'gL': 0.3, 'EL': -65.
})

#set maximum conductance values
max_gNaP = 2
max_gK2 = 2

#increment value
step_gNaP = 1
step_gK2 = 1

connor.pars['gNaP'] = 0
connor.pars['gK2'] = 0
while (connor.pars['gNaP'] <= max_gNaP) and (connor.pars['gK2'] <= max_gK2):

    ## ==============================
	## ANTHONY
	## DON'T CHANGE THINGS BELOW HERE
	## ==============================

	print "Computing trajectories..."
	traj = connor.compute('test')

	print "Sampling trajectories..."
	pts = traj.sample()



	#traj = modButera.compute('test')
	#pts = traj.sample()

	validate= False

	if validate:
		xs = np.arange(-100, 80, 0.1)

		def fn_minf_Butera(v):
			return 1/(1+exp(-(v+34)/5))
		
		def fn_ninf_Butera(v):
			return 1/(1+exp(-(v+29)/4))
		
		def fn_ntau_Butera(v):
			return 10/cosh(-(v+29)/(2*4))   
		
		def fn_mNaPinf_Butera(v):
			return 1/(1+np.exp(-(v+40)/6))
		
		def fn_hNaPinf_Butera(v):
			return 1/(1+np.exp((v+48)/6))
		
		def fn_hNaPtau_Butera(v):
				return 10000/(np.cosh((v+48)/(2*6)))
		
		
		ys_minf_Connor = [connor.auxfns['minf'](v) for v in xs]
		ys_mtau_Connor = [connor.auxfns['mtau'](v) for v in xs]
		ys_hinf_Connor = [connor.auxfns['hinf'](v) for v in xs]
		ys_htau_Connor = [connor.auxfns['htau'](v) for v in xs]
		ys_ninf_Connor = [connor.auxfns['ninf'](v) for v in xs]
		ys_ntau_Connor = [connor.auxfns['ntau'](v) for v in xs]
		ys_minf_Butera = [fn_minf_Butera(v) for v in xs]
		ys_ninf_Butera = [fn_ninf_Butera(v) for v in xs]
		ys_ntau_Butera = [fn_ntau_Butera(v) for v in xs]
		
		plt.figure()
		plt.title('I_Na in/activation')
		plt.plot(xs, ys_minf_Connor, 'b-', label='m, Model')
		plt.plot(xs, ys_minf_Butera, 'b--', label='m, Butera')
		plt.plot(xs, ys_hinf_Connor, 'r-', label='h, Model')
		plt.plot(xs, ys_ninf_Connor, 'g-', label='n, Model')
		plt.plot(xs, ys_ninf_Butera, 'g--', label='n, Butera')
		#plt.plot(xs, ys_hinf_Butera, label='h, Model')
		plt.legend()
		
		
		ys_mNaPinf_Hill = [connor.auxfns['mNaPinf'](v) for v in xs]    
		#ys_mNaPtau_Hill = [connor.auxfns['mNaPtau'](v) for v in xs]
		ys_mNaPinf_Butera = [fn_mNaPinf_Butera(v) for v in xs]
		ys_hNaPinf_Butera = [fn_hNaPinf_Butera(v) for v in xs]
		ys_hNaPtau_Butera = [fn_hNaPtau_Butera(v) for v in xs]
		
		plt.figure()
		plt.title('I_NaP in/activation')
		plt.plot(xs, ys_mNaPinf_Hill, 'b-', label='mNaP, Model')
		plt.plot(xs, ys_mNaPinf_Butera, 'b--', label='mNaP, Butera')
		plt.plot(xs, ys_hNaPinf_Butera, 'g--', label='hNaP, Butera')
		plt.legend()
		
		plt.figure()
		plt.title('I_NaP time scales')
		#plt.plot(xs, ys_mNaPtau_Hill, 'b-', label='mNaP, Model')
		plt.plot(xs, ys_hNaPtau_Butera, 'g--', label='hNaP, Butera')
		plt.legend()
		
		ys_mK2inf_Hill = [connor.auxfns['mK2inf'](v) for v in xs]
		ys_mK2tau_Hill = [connor.auxfns['mK2tau'](v) for v in xs]
		
		plt.figure()
		plt.title('I_K2 in/activation')
		plt.plot(xs, ys_mK2inf_Hill, label='mK2, Model')
		plt.legend()
		
		plt.figure()
		plt.title('I_K2 time scales')
		plt.plot(xs, ys_mK2tau_Hill, label='mK2, Model')
		plt.legend()
		
		

	#1/0
	#plt.figure()
	#plt.title('In/Activation Variables')
	#plt.plot(pts['t'], pts['m']**3, label='m')
	#plt.plot(pts['t'], pts['h'], label='h')
	#plt.plot(pts['t'], pts['n']**4, label='n')
	#plt.plot(pts['t'], pts['A']**4, label='A')
	#plt.plot(pts['t'], pts['B'], label='B')
	##plt.plot(pts['t'], pts['mNaP'], label='m_NaP')
	#plt.plot(pts['t'], pts['hNaP'], label='h_NaP')
	#plt.plot(pts['t'], pts['mK2']**2, label='m_K2')
	#plt.legend()

	#plt.figure()
	#plt.title('Ionic & Stimulus Currents')
	#plt.plot(pts['t'], pts['Iext'], label='I_ext')

	#plt.plot(pts['t'], pts['INa'], label='I_Na')
	#plt.plot(pts['t'], pts['IK'], label='I_K')
	#plt.plot(pts['t'], pts['ILeak'], label='I_Leak')
	#plt.plot(pts['t'], pts['IA'], label='I_A')
	#plt.plot(pts['t'], pts['INaP'], label='I_NaP')
	#plt.plot(pts['t'], pts['IK2'], label='I_K2')
	#plt.legend()

	plt.figure()
	plt.title('Membrane Potential')
	plt.plot(pts['t'], pts['v'])
	plt.ylim([-100,80])


	plt.draw()
	plt.show()

	num = connor.pars['gNaP']
	bum = connor.pars['gK2']
	plt.savefig('C:\Users\Churp\Pictures\(%d,%d).png' % (num,bum))
	"""Saves graph and names according to (gNaP, gK2)"""

	plt.close()
	"""Closes Graph"""
	connor.pars['gNaP'] = connor.pars['gNaP'] + step_gNaP
	if connor.pars['gNaP'] > max_gNaP:
		connor.pars['gNaP'] = 0
		connor.pars['gK2'] = connor.pars['gK2'] + step_gK2