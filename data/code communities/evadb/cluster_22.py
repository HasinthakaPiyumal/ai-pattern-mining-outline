# Cluster 22

class EvaDBServerTest(unittest.IsolatedAsyncioTestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @patch('evadb.evadb_server.start_evadb_server')
    @patch('asyncio.run')
    def test_main(self, mock_run, mock_start_evadb_server):
        main()
        mock_start_evadb_server.assert_called_once()
        mock_run.assert_called_once()

    @patch('evadb.evadb_server.start_evadb_server')
    @patch('asyncio.start_server')
    async def test_start_evadb_server(self, mock_start_evadb_server, mock_start):
        await start_evadb_server(EvaDB_DATABASE_DIR, '0.0.0.0', 8803)
        mock_start_evadb_server.assert_called_once()

def main():
    parser = argparse.ArgumentParser(description='EvaDB Server')
    parser.add_argument('--host', help='Specify the host address on which the server will start.')
    parser.add_argument('--port', help='Specify the port number on which the server will start.')
    parser.add_argument('--db_dir', help='Specify the evadb directory which the server should access.')
    parser.add_argument('--sql_backend', help='Specify the custom sql database to use for structured data.')
    parser.add_argument('--start', help='start server', action='store_true', default=True)
    parser.add_argument('--stop', help='stop server', action='store_true', default=False)
    args, unknown = parser.parse_known_args()
    args.host = args.host or '0.0.0.0'
    args.port = args.port or '8803'
    if args.stop:
        return stop_server()
    if args.start:
        asyncio.run(start_evadb_server(args.db_dir, args.host, args.port, args.sql_backend))

def stop_server():
    """
    Stop the evadb server
    """
    for proc in process_iter():
        if proc.name() == 'evadb_server':
            proc.send_signal(SIGTERM)
    return 0

