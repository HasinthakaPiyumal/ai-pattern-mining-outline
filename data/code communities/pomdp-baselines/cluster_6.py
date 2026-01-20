# Cluster 6

class HumanOutputFormat(KVWriter, SeqWriter):

    def __init__(self, filename_or_file):
        if isinstance(filename_or_file, str):
            self.file = open(filename_or_file, 'wt')
            self.own_file = True
        else:
            assert hasattr(filename_or_file, 'read'), 'expected file or str, got %s' % filename_or_file
            self.file = filename_or_file
            self.own_file = False

    def writekvs(self, kvs):
        key2str = {}
        for key, val in sorted(kvs.items()):
            if isinstance(val, float):
                valstr = '%-8.3g' % (val,)
            else:
                valstr = str(val)
            key2str[self._truncate(key)] = self._truncate(valstr)
        if len(key2str) == 0:
            print('WARNING: tried to write empty key-value dict')
            return
        else:
            keywidth = max(map(len, key2str.keys()))
            valwidth = max(map(len, key2str.values()))
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f %Z')
        dashes = '-' * (keywidth + valwidth + 7)
        dashes_time = put_in_middle(dashes, timestamp)
        lines = [dashes_time]
        for key, val in sorted(key2str.items()):
            lines.append('| %s%s | %s%s |' % (key, ' ' * (keywidth - len(key)), val, ' ' * (valwidth - len(val))))
        lines.append(dashes)
        self.file.write('\n'.join(lines) + '\n')
        self.file.flush()

    def _truncate(self, s):
        return s[:30] + '...' if len(s) > 33 else s

    def writeseq(self, seq):
        for arg in seq:
            self.file.write(arg + ' ')
        self.file.write('\n')
        self.file.flush()

    def close(self):
        if self.own_file:
            self.file.close()

def put_in_middle(str1, str2):
    n = len(str1)
    m = len(str2)
    if n <= m:
        return str2
    else:
        start = (n - m) // 2
        return str1[:start] + str2 + str1[start + m:]

