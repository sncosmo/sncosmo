import math

def value_error_str(value, error, latex=False):
    """Return a string representing value and uncertainty.

    If latex=True, use '\pm' and '\times'.
    """
    pm = '\pm' if latex else '+/-'
    
    first = int(math.floor(math.log10(abs(value))))  # first significant digit
    last = int(math.floor(math.log10(error)))  # last significant digit

    # use exponential notation if
    # value > 1000 and error > 1000 or value < 0.01
    if (first > 2 and last > 2) or first < -2:
        value /= 10**first
        error /= 10**first
        p = max(0, first - last + 1)
        result = (('({:.' + str(p) + 'f} {:s} {:.'+ str(p) + 'f})')
                  .format(value, pm, error))
        if latex:
            result += ' \\times 10^{{{:d}}}'.format(first)
        else:
            result += ' x 10^{:d}'.format(first)
        return result
    else:
        p = max(0, -last + 1)
        return (('{:.' + str(p) + 'f} {:s} {:.'+ str(p) + 'f}')
                .format(value, pm, error))
