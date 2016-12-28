/*
   evalute some things about salt2 model for testing purposes.
*/

#include <string>
#include <vector>
#include <fstream>

#include "fileutils.h"
#include "fullfit.h"
#include "measureddata.h"
#include "pathutils.h"
#include "instrument.h"
#include "saltmodel.h"
#include "salt2model.h"
#include "saltnirmodel.h"
#include "snfitexception.h"

using namespace std;

double phasemin=-15;
double phasemax=+45;
double salt1_default_wmin = 3460;
double salt1_default_wmax = 6600;
double salt2_default_wmin = 3000;
double salt2_default_wmax = 7000;
//double uband_calibration_error = 0.1;
bool apply_zp_error=true;
//map< string,vector<string> > fixableparameters;


#define WITHEXCEPTIONS

//#define WITH_SPECTRA

#define NBANDS_FOR_SINGLE_FIT 1


#define _GNU_SOURCE 1
#ifndef __USE_GNU
#define __USE_GNU
#endif
#include <fenv.h>

/* copied from LcForFit::ComputeWeightMatForFit and modified to just compute
 * model covariance for specified dates and filters */
static double sqr(const double &x) { return x*x;}

void model_rcov(Salt2Model& model, double *times, size_t npoints,
                Filter *filter,
                bool use_model_errors, 
                bool use_kcorr_errors,
                Mat& ModelCovMat) {
  
    ModelCovMat.Zero();
  
    //cout <<"phase_weight #3 = " << phase_weight << endl;
    // on rajoute des trucs
    if(use_model_errors || use_kcorr_errors) {

        const double * d = times;
        double * w  = ModelCovMat.NonConstData();
    
        for(size_t c=0;c<npoints;c++,d++,w+=(npoints+1)) {
// *w = ModelCovMat(c,c)
            
            if(use_model_errors) {
                // add to error squared to diagonal
                *w += sqr(model.ModelRelativeError(filter,*d,true));
            }
        }
    

        if (use_kcorr_errors) {
            // now add kcorr errors
            double kvar = sqr(model.ModelKError(filter));
            w = ModelCovMat.NonConstData();
            for(size_t c=0; c<npoints*npoints; c++, w++) {
                *w += kvar;
            }
        }
    }
}

vector<double> read_1d_vector(const string &fname)
{
    vector<double> result;
    string line;
    ifstream in(fname.c_str());
    while (getline(in, line))
        result.push_back(atof(line.c_str()));
    return result;
}

int main(int nargs, char **args)
{
    // Grid2DFunction testing
    Grid2DFunction f("../sncosmo/tests/data/interpolation_test_input.dat");
    vector<double> x = read_1d_vector("../sncosmo/tests/data/interpolation_test_evalx.dat");
    vector<double> y = read_1d_vector("../sncosmo/tests/data/interpolation_test_evaly.dat");
    ofstream interp_out("../sncosmo/tests/data/interpolation_test_result.dat");
    for (size_t i=0; i < x.size(); i++) {
        cerr << x[i] << endl;
        for (size_t j=0; j < y.size(); j++) {
            interp_out << f.Value(x[i], y[j]);
            if (j != y.size() - 1) interp_out << " ";
        }
        interp_out << endl;
    }
    interp_out.close();

    // Load model
    Salt2Model *model=new Salt2Model();
    model->SetWavelengthRange(salt2_default_wmin,salt2_default_wmax);

    // Print parameters
    cerr << model->params << endl;

    // print a spectrum
    ofstream specfile("testspec.dat");
    for (double w = 2500.0; w < 9200.; w++) {
        specfile << w << " " << model->SpectrumFlux(w, 0.) << endl;
    }
    specfile.close();

    // Set model parameters to result of fit from SDSS19230

    model->params.LocateParam("DayMax")->val = 5.43904106e+04;
    model->params.LocateParam("Redshift")->val = 2.21000000e-01;
    model->params.LocateParam("Color")->val = 7.95355053e-02;
    model->params.LocateParam("X0")->val = 5.52396533e-05;
    model->params.LocateParam("X1")->val = -1.62106971e+00;
    
    // getting model errors:
    Instrument *instrument = new Instrument("SDSS");
    Filter g = instrument->EffectiveFilterByBand("g");
    Filter r = instrument->EffectiveFilterByBand("r");
    Filter i = instrument->EffectiveFilterByBand("i");
    cerr << "g mean: " << g.Mean() << endl;
    cerr << "r mean: " << r.Mean() << endl;
    cerr << "i mean: " << i.Mean() << endl;
    cerr << model->ModelKError(&g) << endl;
    cerr << model->ModelKError(&r) << endl;
    cerr << model->ModelKError(&i) << endl;
    // model->ModelRelativeError(Filter, double (observerday), bool (with_scaling))

    // just some times with no particular significance.
    double times[20] = {54346.219, 54356.262, 54358.207, 54359.172, 54365.238,
                     54373.313, 54382.246, 54386.25,  54388.254, 54393.238,
                     54403.168, 54406.16,  54412.16,  54416.156, 54420.184,
                     54421.164, 54423.156, 54425.156, 54431.164, 54433.16};
    size_t npoints = 20;
    Mat modelrcov(npoints, npoints);
    model_rcov(*model, times, npoints, &g, true, false, modelrcov);
    // cerr << modelrcov << endl;
    double * rcov = modelrcov.NonConstData();
    for (size_t i=0; i < npoints; i++, rcov+=(npoints+1)) {
        cerr << i << "  " << times[i] << "    " << *rcov << endl;
    }

}
