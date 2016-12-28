# Miscellaneous Development Scripts

## SALT2 test helper

`test_salt2model.cc` is a C++ program that outputs some test data from the snfit implementation of the SALT2 model (for testing against as a reference implementation). It links to a built snfit library. To build (on Linux anyway), create a file `Make.user` containing two variables: `LIB` and `INC`. These should point to the directories containing the built `snfit` library and the snfit header files. Example:

```
INC=/path/to/snfit/install/dir/include
LIB=/path/to/snfit/install/dir/lib
```

Then run `make`.

The file `gen_interp_test_data.py` generates files in `sncosmo/tests/data` that
the C++ program reads.


## `gen_example_data.py`

This is used to, as the name implies, generate the example photometric data distributed with sncosmo and loaded with `load_example_data.py`.

