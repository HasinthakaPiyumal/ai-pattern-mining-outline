# Cluster 23

def main():
    """main function start server use uvicorn default workers: 3 default port:

    23333.
    """
    HuixiangDouEnv.print_env()
    uvicorn.run('web.main:app', host='0.0.0.0', port=int(SERVER_PORT), timeout_keep_alive=600, workers=3, log_config=LOGGING_CONFIG)

