# Miscellaneous Development Scripts

## SALT2 test helper

`test_salt2model.cc` is a C++ program that outputs some test data from the snfit implementation of the SALT2 model (for testing against as a reference implementation). It links to a built snfit library. To build (on Linux anyway), create a file `Make.user` containing two variables: `LIB` and `INC`. These should point to the directories containing the built `snfit` library and the snfit header files. Example:

```
INC=/path/to/snfit/install/dir/include
LIB=/path/to/snfit/install/dir/lib
```

Then run `make`. Set the `SALTPATH` environment variable to the location
of the SALT data files before running the program. In bash:

```
export SALTPATH=/path/to/snfit_data
```


## `gen_interp_test_data.py`

Generates files in `sncosmo/tests/data` that the above `test_salt2model` C++
program reads.


## `gen_example_data.py`

This is used, as the name implies, to generate the example photometric data distributed with sncosmo and loaded with `load_example_data.py`.
