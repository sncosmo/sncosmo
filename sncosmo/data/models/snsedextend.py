#! /usr/bin/env python
#S.rodney
# 2011.05.04
"""
Extrapolate the Hsiao SED down to 300 angstroms
to allow the W filter to reach out to z=2.5 smoothly
in the k-correction tables
"""

import os
from numpy import *
from pylab import * 

sndataroot = os.environ['SNDATA_ROOT']

MINWAVE = 300     # min wavelength for extrapolation (Angstroms)
MAXWAVE = 18000   # max wavelength for extrapolation (Angstroms)

def mkSALT2_UV2IR( showplots=False ) : 
    """ do all the extrapolations needed to extend the SALT2 
    model deep into the UV and the IR (300 to 25000 angstroms)
    and out to +100 days after peak 
    """
    import shutil
    indir = os.path.join( sndataroot, 'models/SALT2/SALT2.Guy10_LAMOPEN' )
    outdir = os.path.join( sndataroot, 'models/SALT2/SALT2.Guy10_UV2IR' )
    
    indat = os.path.join(indir,'salt2_color_correction.dat')
    outdat = os.path.join(outdir,'salt2_color_correction.dat')
    if not os.path.isfile( outdat ) : shutil.copy( indat, outdat ) 

    outinfo = os.path.join(outdir,'SALT2.INFO')
    fout = open(outinfo,'w')
    print >> fout, """
# open rest-lambda range WAAAY beyond nominal 2900-7000 A range.
RESTLAMBDA_RANGE:  300. 25000.
COLORLAW_VERSION: 1
COLORCOR_PARAMS: 2800 7000 4 -0.537186 0.894515 -0.513865 0.0891927
COLOR_OFFSET:  0.0  

MAG_OFFSET: 0.27
SEDFLUX_INTERP_OPT: 1  # 1=>linear,    2=>spline
ERRMAP_INTERP_OPT:  1  # 0=snake off;  1=>linear  2=>spline
ERRMAP_KCOR_OPT:    1  # 1/0 => on/off

MAGERR_FLOOR:   0.005            # don;t allow smaller error than this
MAGERR_LAMOBS:  0.1  2000  4000  # magerr minlam maxlam
MAGERR_LAMREST: 0.1   100   200  # magerr minlam maxlam
"""

    extendSALT2_temp0( salt2dir = 'models/SALT2/SALT2.Guy10_UV2IR', 
                       tailsedfile = 'snsed/Hsiao07.extrap.dat',
                       wjoinblue = 2800, wjoinred = 8500 ,
                       wmin = 300, wmax = 25000, tmin=-20, tmax=100, 
                       showplots=showplots )

    extendSALT2_temp1( salt2dir = 'models/SALT2/SALT2.Guy10_UV2IR', 
                       wjoinblue = 2000, wjoinred = 8500 ,
                       wmin = 300, wmax = 25000, tmin=-20, tmax=100, 
                       wstep = 10, showplots=showplots )

    for sedfile in ['salt2_lc_dispersion_scaling.dat',
                    'salt2_lc_relative_covariance_01.dat',
                    'salt2_lc_relative_variance_0.dat',
                    'salt2_lc_relative_variance_1.dat',
                    'salt2_spec_covariance_01.dat',
                    'salt2_spec_variance_0.dat',
                    'salt2_spec_variance_1.dat' ] : 
        indat = os.path.join( indir, sedfile )
        outdat = os.path.join( outdir, sedfile )
        extrapolatesed_flatline( indat, outdat, showplots=showplots )


def getsed( sedfile = os.path.join( sndataroot, 'snsed/Hsiao07.dat')  ) : 
    d,w,f = loadtxt( sedfile, unpack=True ) 

    #d = d.astype(int)
    days = unique( d ) 

    dlist = [ d[ where( d == day ) ] for day in days ]
    wlist = [ w[ where( d == day ) ] for day in days ]
    flist = [ f[ where( d == day ) ] for day in days ]

    return( dlist, wlist, flist )

def plotsed( sedfile= os.path.join( sndataroot, 'snsed/Hsiao07.dat'), 
             day='all', normalize=False, **kwarg): 
    dlist,wlist,flist = getsed( sedfile ) 
    #days = unique( dlist ) 
    for i in range( len(wlist) ) : 
        thisday = dlist[i][0]

        #defaults = { 'label':str(thisday) } 
        #plotarg = dict( kwarg.items() + defaults.items() )
        if day!='all' : 
            if abs(thisday-day)>0.6 : continue
        if normalize : 
            plot( wlist[i], flist[i]/flist[i].max()+thisday, **kwarg )
        else : 
            plot( wlist[i], flist[i], label=str(thisday), **kwarg )
        # user_in=raw_input('%i : return to continue'%i)


def extrapolatesed_linear(sedfile, newsedfile, minwave=MINWAVE, maxwave=MAXWAVE, Npt=2,tmin=-20, tmax=100, showplots=False ):
    """ use a linear fit of the first/last Npt  points on the SED
    to extrapolate  """

    from scipy import interpolate as scint
    from scipy import stats
    import shutil
    
    dlist,wlist,flist = getsed( sedfile ) 
    dlistnew, wlistnew, flistnew = [],[],[]

    fout = open( newsedfile, 'w' )
    for i in range( len(dlist) ) : 
        d,w,f = dlist[i],wlist[i],flist[i]

        wavestep = w[1] - w[0]
        # blueward linear extrapolation from first N points
        wN = w[:Npt]
        fN = f[:Npt]
        (a,b,rval,pval,stderr)=stats.linregress(wN,fN)

        Nbluestep = len( arange( minwave, w[0], wavestep ) )
        wextBlue = sorted( [ w[0] -(i+1)*wavestep for i in range(Nbluestep) ] )
        fextBlue = array( [ max( 0, a * wave + b ) for wave in wextBlue ] )

        # redward linear extrapolation from first N points
        wN = w[-Npt:]
        fN = f[-Npt:]
        (a,b,rval,pval,stderr)=stats.linregress(wN,fN)

        Nredstep = len( arange( w[-1], maxwave,  wavestep ) )
        wextRed =  sorted( [ w[-1] + (i+1)*wavestep for i in range(Nredstep) ] )
        fextRed = array( [ max( 0, a * wave + b ) for wave in wextRed ] )

        wnew = append( append( wextBlue, w ), wextRed )
        fnew = append( append( fextBlue, f ), fextRed )
        # dnew = zeros( len(wnew) ) + d[0]
        
        for i in range( len( wnew ) ) :
            print >> fout, "%5.1f  %10i  %12.7e"%( d[0], wnew[i], fnew[i] )
    fout.close() 

    return( newsedfile )


def extrapolatesed_flatline(sedfile, newsedfile, minwave=MINWAVE, maxwave=MAXWAVE, tmin=-20, tmax=100, showplots=False ):
    """ use a linear fit of the first/last Npt  points on the SED
    to extrapolate  """

    from scipy import interpolate as scint
    from scipy import stats
    import shutil
    
    dlist,wlist,flist = getsed( sedfile ) 
    dlistnew, wlistnew, flistnew = [],[],[]
    olddaylist = [ round(d) for d in unique(ravel(array(dlist))) ]

    fout = open( newsedfile, 'w' )
    newdaylist = range( tmin, tmax+1 )
    fmed = []
    for thisday in newdaylist : 
        if thisday in olddaylist : 
            iday = olddaylist.index( thisday )
            d,w,f = dlist[iday],wlist[iday],flist[iday]
            wavestep = w[1] - w[0]

            # blueward flatline extrapolation from first point
            Nbluestep = len( arange( minwave, w[0], wavestep ) )
            wextBlue = sorted( [ w[0] -(i+1)*wavestep for i in range(Nbluestep) ] )
            fextBlue = array( [ f[0] for wave in wextBlue ] )

            # redward flatline extrapolation from last point
            Nredstep = len( arange( w[-1], maxwave,  wavestep ) )
            wextRed =  sorted( [ w[-1] + (i+1)*wavestep for i in range(Nredstep) ] )
            fextRed = array( [ f[-1]  for wave in wextRed ] )

            wnew = append( append( wextBlue, w ), wextRed )
            fnew = append( append( fextBlue, f ), fextRed )
            fmed.append( median(f) )
        else : 
            fscaleperday = median( array(fmed[-19:]) / array(fmed[-20:-1]) )
            fnew = fnew * fscaleperday**(thisday-thisdaylast)
        if showplots  :
            clf()
            plot( w, f, 'r-' )
            plot( wnew, fnew, 'k--' )
            ax = gca()
            rcParams['text.usetex']=False
            text(0.95,0.95,'%s\nDay=%i'%(os.path.basename(newsedfile),thisday),ha='right',va='top',transform=ax.transAxes )
            draw()
            userin = raw_input('return to continue')
        for i in range( len( wnew ) ) :
            print >> fout, "%5.1f  %10i  %12.7e"%( thisday, wnew[i], fnew[i] )
        thisdaylast = thisday
    fout.close() 

    return( newsedfile )


def extendNon1a():
    import glob
    import shutil
    sedlist = glob.glob("non1a/SED_NOEXTRAP/*.SED")

    for sedfile in sedlist : 
        newsedfile =  'non1a/' + os.path.basename( sedfile )

        print("EXTRAPOLATING %s"%sedfile)
        extrapolatesed_linear(sedfile, newsedfile, minwave=MINWAVE, maxwave=MAXWAVE, tmin=-20, tmax=100, Npt=2 )
        print("     Done with %s.\a\a\a"%sedfile)





def extendSALT2_temp0( salt2dir = 'models/SALT2/SALT2.Guy10_UV2IR', 
                       tailsedfile = 'snsed/Hsiao07.extrap.dat',
                       wjoinblue = 2800, wjoinred = 8500 ,
                       wmin = 300, wmax = 25000, tmin=-20, tmax=100, 
                       showplots=False ):
    """ extend the salt2 Template_0 model component 
    by adopting the UV and IR tails from another SED model. 
    The default is to use SR's extrapolated modification 
    of the Hsiao 2007 sed model, scaled and joined at the 
    wjoin wavelengths, and extrapolated out to wmin and wmax. 
    """
    import shutil
    sndataroot = os.environ['SNDATA_ROOT']
 
    salt2dir = os.path.join( sndataroot, salt2dir ) 
    
    temp0fileIN = os.path.join( salt2dir, '../SALT2.Guy10_LAMOPEN/salt2_template_0.dat' ) 
    temp0fileOUT = os.path.join( salt2dir, 'salt2_template_0.dat' ) 
    temp0dat = getsed( sedfile=temp0fileIN ) 

    tailsedfile = os.path.join( sndataroot, tailsedfile ) 

    taildat = getsed( sedfile=tailsedfile ) 
    
    dt,wt,ft = loadtxt( tailsedfile, unpack=True ) 
    taildays = unique( dt ) 

    fscale = []
    # build up modified template from day -20 to +100
    outlines = []
    daylist = range( tmin, tmax+1 )
    for i in range( len(daylist) ) : 
        thisday = daylist[i]

        if thisday < 50 : 
            # get the tail SED for this day from the Hsiao template
            it = where( taildays == thisday )[0]
            dt = taildat[0][it]
            wt = taildat[1][it]
            ft = taildat[2][it]

            # get the SALT2 template SED for this day
            d0 = temp0dat[0][i]
            w0 = temp0dat[1][i]
            f0 = temp0dat[2][i]
            print( 'splicing tail onto template for day : %i'%thisday )

            i0blue = argmin(  abs(w0-wjoinblue) )
            itblue = argmin( abs( wt-wjoinblue))

            i0red = argmin(  abs(w0-wjoinred) )
            itred = argmin( abs( wt-wjoinred))

            itmin = argmin( abs( wt-wmin))
            itmax = argmin( abs( wt-wmax))

            bluescale = f0[i0blue]/ft[itblue] 
            redscale = f0[i0red]/ft[itred] 

            d0new = dt.tolist()[itmin:itblue] + d0.tolist()[i0blue:i0red] + dt.tolist()[itred:itmax+1]
            w0new = wt.tolist()[itmin:itblue] + w0.tolist()[i0blue:i0red] + wt.tolist()[itred:itmax+1]
            f0newStage = (bluescale*ft).tolist()[itmin:itblue] + f0.tolist()[i0blue:i0red] + (redscale*ft).tolist()[itred:itmax+1]

            # compute the flux scaling decrement from the last epoch (for extrapolation)
            if i>1: fscale.append( np.where( np.array(f0newStage)<=0, 0, ( np.array(f0newStage) / np.array(f0new) ) ) )
            f0new = f0newStage

        # elif thisday < 85 : 
        #     # get the full SED for this day from the Hsiao template
        #     it = where( taildays == thisday )[0]
        #     dt = taildat[0][it]
        #     wt = taildat[1][it]
        #     ft = taildat[2][it]
        #     d0new = dt
        #     w0new = wt
        #     f0new = ft * (bluescale+redscale)/2.  * (fscaleperday**(thisday-50))
        else  : 
            print( 'scaling down last template to extrapolate to day : %i'%thisday )
            # linearly scale down the last Hsiao template
            d0new = zeros( len(dt) ) + thisday
            w0new = wt
            #f0new = f0new * (bluescale+redscale)/2. * (fscaleperday**(thisday-50))

            f0new = np.array(f0new) * np.median( np.array(fscale[-20:]), axis=0 )
            #f0new = np.array(f0new) * ( np.median(fscale[-20:])**(thisday-50))

        if showplots: 
            # plot it
            print( 'plotting modified template for day : %i'%thisday )
            clf()
            plot( w0, f0, ls='-',color='b', lw=1)
            plot( wt, (bluescale+redscale)/2. * ft, ls=':',color='r', lw=1)
            plot( w0new, f0new, ls='--',color='k', lw=2)
            ax = gca()
            ax.grid()
            ax.set_xlim( 500, 13000 )
            ax.set_ylim( -0.001, 0.02 )
            draw()
            raw_input('return to continue')

        # append to the list of output data lines
        for j in range( len( d0new ) ) :
            outlines.append( "%6.2f    %12i  %12.7e\n"%(
                    d0new[j], w0new[j], f0new[j] ) )

    # write it out to the new template sed .dat file
    fout = open( temp0fileOUT, 'w' ) 
    fout.writelines( outlines ) 
    fout.close() 
        


def extendSALT2_temp1( salt2dir = 'models/SALT2/SALT2.Guy10_UV2IR', 
                       wjoinblue = 2000, wjoinred = 8500 ,
                       wmin = 300, wmax = 25000, tmin=-20, tmax=100, 
                       wstep = 10, showplots=False ):
    """ extend the salt2 Template_1 model component 
    with a flat line at 0 to the blue and to the red.
    """
    import shutil
    sndataroot = os.environ['SNDATA_ROOT']
 
    salt2dir = os.path.join( sndataroot, salt2dir ) 
    
    temp1fileIN = os.path.join( salt2dir, '../SALT2.Guy10_LAMOPEN/salt2_template_1.dat' ) 
    temp1fileOUT = os.path.join( salt2dir, 'salt2_template_1.dat' ) 
    temp1dat = getsed( sedfile=temp1fileIN ) 

    # build up modified template from day -20 to +100
    outlines = []
    daylist = range( tmin, tmax+1 )
    f1med = []
    for i in range( len(daylist) ) : 
        thisday = daylist[i]

        if thisday < 50 : 
            print( 'extrapolating with flatline onto template for day : %i'%thisday )
            # get the SALT2 template SED for this day
            d1 = temp1dat[0][i]
            w1 = temp1dat[1][i]
            f1 = temp1dat[2][i]
            
            i1blue = argmin(  abs(w1-wjoinblue) )
            i1red = argmin(  abs(w1-wjoinred) )

            Nblue = (wjoinblue-wmin )/wstep + 1
            Nred = (wmax -wjoinred )/wstep + 1

            d1new =  (ones(Nblue)*thisday).tolist() + d1.tolist()[i1blue+1:i1red] + (ones(Nred)*thisday).tolist()
            w1new = range(wmin,wmin+Nblue*wstep,wstep) + w1.tolist()[i1blue+1:i1red] + range(wjoinred,wjoinred+Nred*wstep,wstep)
            f1new = array( zeros(Nblue).tolist() + f1.tolist()[i1blue+1:i1red] + zeros(Nred).tolist() )
            
            f1med.append( median( f1 ) )

        else : 
            print( 'blind extrapolation for day : %i'%thisday )
            d1new = zeros( len(d1) ) + thisday
            f1scaleperday = median(array(f1med[-19:]) / array(f1med[-20:-1]) )
            f1new = array( zeros(Nblue).tolist() + f1.tolist()[i1blue+1:i1red-1] + zeros(Nred).tolist() )
            f1new = f1new * f1scaleperday**(thisday-50)

        if showplots and thisday>45:
            # plot it
            clf()
            plot( w1, f1, ls='-',color='r', lw=1)
            plot( w1new, f1new, ls='--',color='k', lw=2)
            draw()
            raw_input('return to continue')

        # append to the list of output data lines
        for j in range( len( d1new ) ) :
            outlines.append( "%6.2f    %12i  %12.7e\n"%(
                    d1new[j], w1new[j], f1new[j] ) )

    # write it out to the new template sed .dat file
    fout = open( temp1fileOUT, 'w' ) 
    fout.writelines( outlines ) 
    fout.close() 
        


def extendSALT2_flatline( salt2dir = 'models/SALT2/SALT2.Guy10_UV2IR', 
                          wjoinblue = 2000, wjoinred = 8500 ,
                          wmin = 300, wmax = 18000, tmin=-20, tmax=100, 
                          wstep = 10, showplots=False ):
    """ extrapolate the *lc* and *spec* .dat files for SALT2
    using a flatline to the blue and red """

    sndataroot = os.environ['SNDATA_ROOT']
    salt2dir = os.path.join( sndataroot, salt2dir ) 
    
    filelist = ['salt2_lc_dispersion_scaling.dat',
                'salt2_lc_relative_covariance_01.dat',
                'salt2_lc_relative_variance_0.dat',
                'salt2_lc_relative_variance_1.dat',
                'salt2_spec_covariance_01.dat',
                'salt2_spec_variance_0.dat',
                'salt2_spec_variance_1.dat']
    
    #for filename in  ['salt2_lc_dispersion_scaling.dat']: 
    #for filename in  ['salt2_lc_relative_covariance_01.dat']:
    for filename in filelist : 
        infile = os.path.join( salt2dir, 'NO_SED_EXTRAP/' + filename )
        outfile = os.path.join( salt2dir, filename )

        newsedfile = extrapolatesed_flatline( infile, outfile, minwave=wmin, maxwave=wmax, tmin=tmin, tmax=tmax )
        
        # plot it
        if showplots: 
            #for d in range(-20,50) : 
            for d in [-10,-5,0,5,10,15,20,25,30,35,40,45,50,60,70,80,90] : 
                clf()
                plotsed( infile, day=d, ls='-',color='r', lw=1) 
                plotsed( outfile,day=d, ls='--',color='k', lw=2)
                print( '%s : day %i'%(filename,d) )
                draw()
                # raw_input('%s : day %i.  return to continue'%(filename,d)) 


