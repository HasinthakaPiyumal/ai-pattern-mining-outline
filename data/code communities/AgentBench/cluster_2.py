# Cluster 2

@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    orig_out_err = (sys.stdout, sys.stderr)
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    except Exception as exc:
        raise exc
    finally:
        sys.stdout, sys.stderr = orig_out_err

