import os

if os.environ['ISX'] == '1':
    try:
        import isx
    except ImportError as error:
        print('The Inscopix Data Processing Software (IDPS) API is not installed!')
        raise error
