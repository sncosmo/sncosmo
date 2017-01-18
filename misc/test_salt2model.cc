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

Mat model_rcov(Salt2Model& model, const vector<double>& times,
                Filter *filter,
                bool use_model_errors, 
                bool use_kcorr_errors)
{
    double *w;
    size_t npoints = times.size();

    //initialize result
    Mat result(npoints, npoints);
    result.Zero();

    // model errors: added on diagonal only
    if (use_model_errors) {
        w = result.NonConstData();
        for(size_t i=0; i<npoints; i++, w+=(npoints+1)) {
            *w += sqr(model.ModelRelativeError(filter, times[i], true));
        }
    }
    
    // kcorr error added to every element
    if (use_kcorr_errors) {
        double kvar = sqr(model.ModelKError(filter));
        w = result.NonConstData();
        for(size_t i=0; i<npoints*npoints; i++, w++) {
            *w += kvar;
        }
    }

    return result;
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

// write model parameters to "header" of file
void write_model_params(ofstream& out, const Salt2Model *model)
{
    out << "@Redshift " << model->params.LocateParam("Redshift")->val << endl;
    out << "@DayMax " << model->params.LocateParam("DayMax")->val << endl;
    out << "@X0 " << model->params.LocateParam("X0")->val << endl;
    out << "@X1 " << model->params.LocateParam("X1")->val << endl;
    out << "@Color " << model->params.LocateParam("Color")->val << endl;
}


// Evaluate the model at various times and wavelengths and write results to a
// file.
void evaluate_and_write_timeseries(const Salt2Model *model,
                                   const vector<double>& time,
                                   const vector<double>& wave,
                                   const string& fname)
{
    ofstream out(fname.c_str());
    out.setf(ios::showpoint);
    out.precision(7);
    
    // write header
    out << "# output from snfit Salt2Model.SpectrumFlux with following parameters\n";
    write_model_params(out, model);

    for (auto const& t: time) {
        for (auto const& w: wave) {
            out.precision(7);
            out << t << " " << w << " ";
            out.precision(20);  // extra precision on flux
            out << model->SpectrumFlux(w, t) << endl;
        }
    }
    out.close();
}

void test_bandpass_interpolation() {

    Instrument *megacampsf = new Instrument("MEGACAMPSF");

    double area = megacampsf->MirrorArea();
    
    // evaluate two filters at two different radial positions
    vector<string> bands = {"g", "z"};
    vector<double> x_coords = {2.0, 8.0};
    vector<double> y_coords = {3.5, 7.5};
    vector<vector<double>> wave = {
        {3600., 3800., 4000., 4200., 4400., 4600., 4800.,
         5000., 5200., 5400., 5600., 5800., 6000., 6200.},
        {7600., 8000., 8400., 8800., 9200., 9600., 10000.,
         10400., 10800., 11200.}};
    
    for (auto const& band: bands) {
        for (size_t i=0; i<x_coords.size(); i++) {
            
            double x_coord = x_coords[i];
            double y_coord = y_coords[i];
            double r = sqrt(x_coord * x_coord + y_coord * y_coord);
            
            FocalPlanePosition P(x_coord, y_coord);
            Filter filter = megacampsf->EffectiveFilterByBand(band, &P);

            // evaluate filter and write out
            ofstream out("../sncosmo/tests/data/snfit_filter_" + band + "_" + to_string(i) + ".dat");
            out << "@name MEGACAMPSF::" << band << endl;
            out << "@radius " << r;
            for (auto const& w: wave[i]) {
                out.precision(7);
                out << w << " ";
                out.precision(12);
                out << filter.Value(w) / area << endl;
            }
            out.close();
        }
    }
}

    
int main(int nargs, char **args)
{
    // ----------------------------------------------------------------------
    // Grid2DFunction testing
    Grid2DFunction f("../sncosmo/tests/data/interpolation_test_input.dat");
    vector<double> x = read_1d_vector(
        "../sncosmo/tests/data/interpolation_test_evalx.dat");
    vector<double> y = read_1d_vector(
        "../sncosmo/tests/data/interpolation_test_evaly.dat");
    ofstream interp_out("../sncosmo/tests/data/interpolation_test_result.dat");

    for (size_t i=0; i < x.size(); i++) {
        for (size_t j=0; j < y.size(); j++) {
            interp_out << f.Value(x[i], y[j]);
            if (j != y.size() - 1) interp_out << " ";
        }
        interp_out << endl;
    }
    interp_out.close();


    // Load model for model tests below.
    Salt2Model *model=new Salt2Model();
    model->SetWavelengthRange(salt2_default_wmin,salt2_default_wmax);

    //-----------------------------------------------------------------------
    // test spectral surface
    
    // Observer-frame times and wavelengths
    vector<double> time = {-30., -10., 0., 10., 25., 50., 80.};
    vector<double> wave = {3000., 3500., 4000., 4500., 5000., 5500., 6000.,
                           6500., 7000., 7500., 8000., 8500., 9000.};

    // parameters in order: Redshift, daymax, x0, x1, color
    vector<vector<double>> params = {{0.0, 0.0, 1.0, 0.0, 0.0},
                                     {0.15, 0.0, 1.e-5, 0.0, 0.0},
                                     {0.15, 5.0, 1.e-5, 1.0, 0.0},
                                     {0.15, 5.0, 1.e-5, 0.0, 0.1}};
    vector<string> fnames = {"salt2_timeseries_1.dat",
                             "salt2_timeseries_2.dat",
                             "salt2_timeseries_3.dat",
                             "salt2_timeseries_4.dat"};
    for (size_t i = 0; i < params.size(); i++) {
        model->params.LocateParam("Redshift")->val = params[i][0];
        model->params.LocateParam("DayMax")->val = params[i][1];
        model->params.LocateParam("X0")->val = params[i][2];
        model->params.LocateParam("X1")->val = params[i][3];
        model->params.LocateParam("Color")->val = params[i][4];

        evaluate_and_write_timeseries(model, time, wave,
                                      "../sncosmo/tests/data/" + fnames[i]);
    }

    // -----------------------------------------------------------------------
    // Test model relative covariance

    model->params.LocateParam("Redshift")->val = 0.15;
    model->params.LocateParam("DayMax")->val = 6.5;
    model->params.LocateParam("X0")->val = 1.e-5;
    model->params.LocateParam("X1")->val = -1.0;
    model->params.LocateParam("Color")->val = 0.1;
    
    // get some filters
    Instrument *instrument = new Instrument("SDSS");
    vector<Filter> filters = {instrument->EffectiveFilterByBand("g"),
                              instrument->EffectiveFilterByBand("r"),
                              instrument->EffectiveFilterByBand("i")};

    // get some times
    vector<double> times = {-30.0, -20.0, -10.0, 0.0, 10., 20., 30., 40.,
                            50., 80.};

    // write params and times
    ofstream out("../sncosmo/tests/data/salt2_rcov_params_times.dat");
    out.setf(ios::showpoint);
    out.precision(7);
    out << "# parameters and rest-frame times to evaluate model relative covariance\n";
    write_model_params(out, model);
    for (auto const& t: times)
        out << t << " ";
    out << endl;
    out.close();

    // separate results for each filter
    for (size_t i=0; i<filters.size(); i++) {
        Mat modelrcov = model_rcov(*model, times, &filters[i], true, true);
        modelrcov.writeASCII("../sncosmo/tests/data/salt2_rcov_snfit_" + filters[i].InstrumentName() + filters[i].Band() + ".dat");
    }

    // -----------------------------------------------------------------------
    // Test bandpass interpolation

    test_bandpass_interpolation();
    
} // end main
