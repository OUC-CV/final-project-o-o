import matlab.engine

def wlsFilter(IN, lambda_val=0.5, alpha_val=1.2):
    eng = matlab.engine.start_matlab()
    OUT = eng.wlsFilter(matlab.double(IN.tolist()), lambda_val, alpha_val)
    eng.quit()
    return OUT