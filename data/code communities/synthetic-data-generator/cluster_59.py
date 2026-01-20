# Cluster 59

def main():
    """CLI."""
    args = _parse_args()
    if args.tsv:
        data, discrete_columns = read_tsv(args.data, args.metadata)
    else:
        data, discrete_columns = read_csv(args.data, args.metadata, args.header, args.discrete)
    if args.load:
        model = CTGAN.load(args.load)
    else:
        generator_dim = [int(x) for x in args.generator_dim.split(',')]
        discriminator_dim = [int(x) for x in args.discriminator_dim.split(',')]
        model = CTGAN(embedding_dim=args.embedding_dim, generator_dim=generator_dim, discriminator_dim=discriminator_dim, generator_lr=args.generator_lr, generator_decay=args.generator_decay, discriminator_lr=args.discriminator_lr, discriminator_decay=args.discriminator_decay, batch_size=args.batch_size, epochs=args.epochs)
    model.fit(data, discrete_columns)
    if args.save is not None:
        model.save(args.save)
    num_samples = args.num_samples or len(data)
    if args.sample_condition_column is not None:
        assert args.sample_condition_column_value is not None
    sampled = model.sample(num_samples, args.sample_condition_column, args.sample_condition_column_value)
    if args.tsv:
        write_tsv(sampled, args.metadata, args.output)
    else:
        sampled.to_csv(args.output, index=False)

def _parse_args():
    parser = argparse.ArgumentParser(description='CTGAN Command Line Interface')
    parser.add_argument('-e', '--epochs', default=300, type=int, help='Number of training epochs')
    parser.add_argument('-t', '--tsv', action='store_true', help='Load data in TSV format instead of CSV')
    parser.add_argument('--no-header', dest='header', action='store_false', help='The CSV file has no header. Discrete columns will be indices.')
    parser.add_argument('-m', '--metadata', help='Path to the metadata')
    parser.add_argument('-d', '--discrete', help='Comma separated list of discrete columns without whitespaces.')
    parser.add_argument('-n', '--num-samples', type=int, help='Number of rows to sample. Defaults to the training data size')
    parser.add_argument('--generator_lr', type=float, default=0.0002, help='Learning rate for the generator.')
    parser.add_argument('--discriminator_lr', type=float, default=0.0002, help='Learning rate for the discriminator.')
    parser.add_argument('--generator_decay', type=float, default=1e-06, help='Weight decay for the generator.')
    parser.add_argument('--discriminator_decay', type=float, default=0, help='Weight decay for the discriminator.')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of input z to the generator.')
    parser.add_argument('--generator_dim', type=str, default='256,256', help='Dimension of each generator layer. Comma separated integers with no whitespaces.')
    parser.add_argument('--discriminator_dim', type=str, default='256,256', help='Dimension of each discriminator layer. Comma separated integers with no whitespaces.')
    parser.add_argument('--batch_size', type=int, default=500, help='Batch size. Must be an even number.')
    parser.add_argument('--save', default=None, type=str, help='A filename to save the trained synthesizer.')
    parser.add_argument('--load', default=None, type=str, help='A filename to load a trained synthesizer.')
    parser.add_argument('--sample_condition_column', default=None, type=str, help='Select a discrete column name.')
    parser.add_argument('--sample_condition_column_value', default=None, type=str, help='Specify the value of the selected discrete column.')
    parser.add_argument('data', help='Path to training data')
    parser.add_argument('output', help='Path of the output file')
    return parser.parse_args()

def read_tsv(data_filename, meta_filename):
    """Read a tsv file."""
    with open(meta_filename) as f:
        column_info = f.readlines()
    column_info_raw = [x.replace('{', ' ').replace('}', ' ').split() for x in column_info]
    discrete = []
    continuous = []
    column_info = []
    for idx, item in enumerate(column_info_raw):
        if item[0] == 'C':
            continuous.append(idx)
            column_info.append((float(item[1]), float(item[2])))
        else:
            assert item[0] == 'D'
            discrete.append(idx)
            column_info.append(item[1:])
    meta = {'continuous_columns': continuous, 'discrete_columns': discrete, 'column_info': column_info}
    with open(data_filename) as f:
        lines = f.readlines()
    data = []
    for row in lines:
        row_raw = row.split()
        row = []
        for idx, col in enumerate(row_raw):
            if idx in continuous:
                row.append(col)
            else:
                assert idx in discrete
                row.append(column_info[idx].index(col))
        data.append(row)
    return (np.asarray(data, dtype='float32'), meta['discrete_columns'])

def read_csv(csv_filename, meta_filename=None, header=True, discrete=None):
    """Read a csv file."""
    data = pd.read_csv(csv_filename, header='infer' if header else None)
    if meta_filename:
        with open(meta_filename) as meta_file:
            metadata = json.load(meta_file)
        discrete_columns = [column['name'] for column in metadata['columns'] if column['type'] != 'continuous']
    elif discrete:
        discrete_columns = discrete.split(',')
        if not header:
            discrete_columns = [int(i) for i in discrete_columns]
    else:
        discrete_columns = []
    return (data, discrete_columns)

def write_tsv(data, meta, output_filename):
    """Write to a tsv file."""
    with open(output_filename, 'w') as f:
        for row in data:
            for idx, col in enumerate(row):
                if idx in meta['continuous_columns']:
                    print(col, end=' ', file=f)
                else:
                    assert idx in meta['discrete_columns']
                    print(meta['column_info'][idx][int(col)], end=' ', file=f)
            print(file=f)

