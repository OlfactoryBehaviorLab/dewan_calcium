import os
import warnings
try:
    if os.environ['ISX'] == '1':
        try:
            import isx
        except ImportError as error:
            print('The Inscopix Data Processing Software (IDPS) API is not installed!')
            raise error
except KeyError as ke:
    warnings.warn('The ISX environment variable is not present, assuming IDPS API is not needed...')

