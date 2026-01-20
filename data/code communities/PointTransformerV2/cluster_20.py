# Cluster 20

class PlyElement(object):
    """
    PLY file element.

    A client of this library doesn't normally need to instantiate this
    directly, so the following is only for the sake of documenting the
    internals.

    Creating a PlyElement instance is generally done in one of two ways:
    as a byproduct of PlyData.read (when reading a PLY file) and by
    PlyElement.describe (before writing a PLY file).

    """

    def __init__(self, name, properties, count, comments=[]):
        """
        This is not part of the public interface.  The preferred methods
        of obtaining PlyElement instances are PlyData.read (to read from
        a file) and PlyElement.describe (to construct from a numpy
        array).

        """
        self._name = str(name)
        self._check_name()
        self._count = count
        self._properties = tuple(properties)
        self._index()
        self.comments = list(comments)
        self._have_list = any((isinstance(p, PlyListProperty) for p in self.properties))

    @property
    def count(self):
        return self._count

    def _get_data(self):
        return self._data

    def _set_data(self, data):
        self._data = data
        self._count = len(data)
        self._check_sanity()
    data = property(_get_data, _set_data)

    def _check_sanity(self):
        for prop in self.properties:
            if prop.name not in self._data.dtype.fields:
                raise ValueError('dangling property %r' % prop.name)

    def _get_properties(self):
        return self._properties

    def _set_properties(self, properties):
        self._properties = tuple(properties)
        self._check_sanity()
        self._index()
    properties = property(_get_properties, _set_properties)

    def _index(self):
        self._property_lookup = dict(((prop.name, prop) for prop in self._properties))
        if len(self._property_lookup) != len(self._properties):
            raise ValueError('two properties with same name')

    def ply_property(self, name):
        return self._property_lookup[name]

    @property
    def name(self):
        return self._name

    def _check_name(self):
        if any((c.isspace() for c in self._name)):
            msg = 'element name %r contains spaces' % self._name
            raise ValueError(msg)

    def dtype(self, byte_order='='):
        """
        Return the numpy dtype of the in-memory representation of the
        data.  (If there are no list properties, and the PLY format is
        binary, then this also accurately describes the on-disk
        representation of the element.)

        """
        return [(prop.name, prop.dtype(byte_order)) for prop in self.properties]

    @staticmethod
    def _parse_multi(header_lines):
        """
        Parse a list of PLY element definitions.

        """
        elements = []
        while header_lines:
            elt, header_lines = PlyElement._parse_one(header_lines)
            elements.append(elt)
        return elements

    @staticmethod
    def _parse_one(lines):
        """
        Consume one element definition.  The unconsumed input is
        returned along with a PlyElement instance.

        """
        a = 0
        line = lines[a]
        if line[0] != 'element':
            raise PlyParseError("expected 'element'")
        if len(line) > 3:
            raise PlyParseError("too many fields after 'element'")
        if len(line) < 3:
            raise PlyParseError("too few fields after 'element'")
        name, count = (line[1], int(line[2]))
        comments = []
        properties = []
        while True:
            a += 1
            if a >= len(lines):
                break
            if lines[a][0] == 'comment':
                comments.append(lines[a][1])
            elif lines[a][0] == 'property':
                properties.append(PlyProperty._parse_one(lines[a]))
            else:
                break
        return (PlyElement(name, properties, count, comments), lines[a:])

    @staticmethod
    def describe(data, name, len_types={}, val_types={}, comments=[]):
        """
        Construct a PlyElement from an array's metadata.

        len_types and val_types can be given as mappings from list
        property names to type strings (like 'u1', 'f4', etc., or
        'int8', 'float32', etc.). These can be used to define the length
        and value types of list properties.  List property lengths
        always default to type 'u1' (8-bit unsigned integer), and value
        types default to 'i4' (32-bit integer).

        """
        if not isinstance(data, _np.ndarray):
            raise TypeError('only numpy arrays are supported')
        if len(data.shape) != 1:
            raise ValueError('only one-dimensional arrays are supported')
        count = len(data)
        properties = []
        descr = data.dtype.descr
        for t in descr:
            if not isinstance(t[1], str):
                raise ValueError('nested records not supported')
            if not t[0]:
                raise ValueError('field with empty name')
            if len(t) != 2 or t[1][1] == 'O':
                if t[1][1] == 'O':
                    if len(t) != 2:
                        raise ValueError('non-scalar object fields not supported')
                len_str = _data_type_reverse[len_types.get(t[0], 'u1')]
                if t[1][1] == 'O':
                    val_type = val_types.get(t[0], 'i4')
                    val_str = _lookup_type(val_type)
                else:
                    val_str = _lookup_type(t[1][1:])
                prop = PlyListProperty(t[0], len_str, val_str)
            else:
                val_str = _lookup_type(t[1][1:])
                prop = PlyProperty(t[0], val_str)
            properties.append(prop)
        elt = PlyElement(name, properties, count, comments)
        elt.data = data
        return elt

    def _read(self, stream, text, byte_order):
        """
        Read the actual data from a PLY file.

        """
        if text:
            self._read_txt(stream)
        elif self._have_list:
            self._read_bin(stream, byte_order)
        else:
            self._data = _np.fromfile(stream, self.dtype(byte_order), self.count)
        if len(self._data) < self.count:
            k = len(self._data)
            del self._data
            raise PlyParseError('early end-of-file', self, k)
        self._check_sanity()

    def _write(self, stream, text, byte_order):
        """
        Write the data to a PLY file.

        """
        if text:
            self._write_txt(stream)
        elif self._have_list:
            self._write_bin(stream, byte_order)
        else:
            self.data.astype(self.dtype(byte_order), copy=False).tofile(stream)

    def _read_txt(self, stream):
        """
        Load a PLY element from an ASCII-format PLY file.  The element
        may contain list properties.

        """
        self._data = _np.empty(self.count, dtype=self.dtype())
        k = 0
        for line in _islice(iter(stream.readline, b''), self.count):
            fields = iter(line.strip().split())
            for prop in self.properties:
                try:
                    self._data[prop.name][k] = prop._from_fields(fields)
                except StopIteration:
                    raise PlyParseError('early end-of-line', self, k, prop)
                except ValueError:
                    raise PlyParseError('malformed input', self, k, prop)
            try:
                next(fields)
            except StopIteration:
                pass
            else:
                raise PlyParseError('expected end-of-line', self, k)
            k += 1
        if k < self.count:
            del self._data
            raise PlyParseError('early end-of-file', self, k)

    def _write_txt(self, stream):
        """
        Save a PLY element to an ASCII-format PLY file.  The element may
        contain list properties.

        """
        for rec in self.data:
            fields = []
            for prop in self.properties:
                fields.extend(prop._to_fields(rec[prop.name]))
            _np.savetxt(stream, [fields], '%.18g', newline='\r\n')

    def _read_bin(self, stream, byte_order):
        """
        Load a PLY element from a binary PLY file.  The element may
        contain list properties.

        """
        self._data = _np.empty(self.count, dtype=self.dtype(byte_order))
        for k in _range(self.count):
            for prop in self.properties:
                try:
                    self._data[prop.name][k] = prop._read_bin(stream, byte_order)
                except StopIteration:
                    raise PlyParseError('early end-of-file', self, k, prop)

    def _write_bin(self, stream, byte_order):
        """
        Save a PLY element to a binary PLY file.  The element may
        contain list properties.

        """
        for rec in self.data:
            for prop in self.properties:
                prop._write_bin(rec[prop.name], stream, byte_order)

    @property
    def header(self):
        """
        Format this element's metadata as it would appear in a PLY
        header.

        """
        lines = ['element %s %d' % (self.name, self.count)]
        for c in self.comments:
            lines.append('comment ' + c)
        lines.extend(list(map(str, self.properties)))
        return '\r\n'.join(lines)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __str__(self):
        return self.header

    def __repr__(self):
        return 'PlyElement(%r, %r, count=%d, comments=%r)' % (self.name, self.properties, self.count, self.comments)

def _lookup_type(type_str):
    if type_str not in _data_type_reverse:
        try:
            type_str = _data_types[type_str]
        except KeyError:
            raise ValueError('field type %r not in %r' % (type_str, _types_list))
    return _data_type_reverse[type_str]

class PlyProperty(object):
    """
    PLY property description.  This class is pure metadata; the data
    itself is contained in PlyElement instances.

    """

    def __init__(self, name, val_dtype):
        self._name = str(name)
        self._check_name()
        self.val_dtype = val_dtype

    def _get_val_dtype(self):
        return self._val_dtype

    def _set_val_dtype(self, val_dtype):
        self._val_dtype = _data_types[_lookup_type(val_dtype)]
    val_dtype = property(_get_val_dtype, _set_val_dtype)

    @property
    def name(self):
        return self._name

    def _check_name(self):
        if any((c.isspace() for c in self._name)):
            msg = 'Error: property name %r contains spaces' % self._name
            raise RuntimeError(msg)

    @staticmethod
    def _parse_one(line):
        assert line[0] == 'property'
        if line[1] == 'list':
            if len(line) > 5:
                raise PlyParseError("too many fields after 'property list'")
            if len(line) < 5:
                raise PlyParseError("too few fields after 'property list'")
            return PlyListProperty(line[4], line[2], line[3])
        else:
            if len(line) > 3:
                raise PlyParseError("too many fields after 'property'")
            if len(line) < 3:
                raise PlyParseError("too few fields after 'property'")
            return PlyProperty(line[2], line[1])

    def dtype(self, byte_order='='):
        """
        Return the numpy dtype description for this property (as a tuple
        of strings).

        """
        return byte_order + self.val_dtype

    def _from_fields(self, fields):
        """
        Parse from generator.  Raise StopIteration if the property could
        not be read.

        """
        return _np.dtype(self.dtype()).type(next(fields))

    def _to_fields(self, data):
        """
        Return generator over one item.

        """
        yield _np.dtype(self.dtype()).type(data)

    def _read_bin(self, stream, byte_order):
        """
        Read data from a binary stream.  Raise StopIteration if the
        property could not be read.

        """
        try:
            return _np.fromfile(stream, self.dtype(byte_order), 1)[0]
        except IndexError:
            raise StopIteration

    def _write_bin(self, data, stream, byte_order):
        """
        Write data to a binary stream.

        """
        _np.dtype(self.dtype(byte_order)).type(data).tofile(stream)

    def __str__(self):
        val_str = _data_type_reverse[self.val_dtype]
        return 'property %s %s' % (val_str, self.name)

    def __repr__(self):
        return 'PlyProperty(%r, %r)' % (self.name, _lookup_type(self.val_dtype))

class PlyListProperty(PlyProperty):
    """
    PLY list property description.

    """

    def __init__(self, name, len_dtype, val_dtype):
        PlyProperty.__init__(self, name, val_dtype)
        self.len_dtype = len_dtype

    def _get_len_dtype(self):
        return self._len_dtype

    def _set_len_dtype(self, len_dtype):
        self._len_dtype = _data_types[_lookup_type(len_dtype)]
    len_dtype = property(_get_len_dtype, _set_len_dtype)

    def dtype(self, byte_order='='):
        """
        List properties always have a numpy dtype of "object".

        """
        return '|O'

    def list_dtype(self, byte_order='='):
        """
        Return the pair (len_dtype, val_dtype) (both numpy-friendly
        strings).

        """
        return (byte_order + self.len_dtype, byte_order + self.val_dtype)

    def _from_fields(self, fields):
        len_t, val_t = self.list_dtype()
        n = int(_np.dtype(len_t).type(next(fields)))
        data = _np.loadtxt(list(_islice(fields, n)), val_t, ndmin=1)
        if len(data) < n:
            raise StopIteration
        return data

    def _to_fields(self, data):
        """
        Return generator over the (numerical) PLY representation of the
        list data (length followed by actual data).

        """
        len_t, val_t = self.list_dtype()
        data = _np.asarray(data, dtype=val_t).ravel()
        yield _np.dtype(len_t).type(data.size)
        for x in data:
            yield x

    def _read_bin(self, stream, byte_order):
        len_t, val_t = self.list_dtype(byte_order)
        try:
            n = _np.fromfile(stream, len_t, 1)[0]
        except IndexError:
            raise StopIteration
        data = _np.fromfile(stream, val_t, n)
        if len(data) < n:
            raise StopIteration
        return data

    def _write_bin(self, data, stream, byte_order):
        """
        Write data to a binary stream.

        """
        len_t, val_t = self.list_dtype(byte_order)
        data = _np.asarray(data, dtype=val_t).ravel()
        _np.array(data.size, dtype=len_t).tofile(stream)
        data.tofile(stream)

    def __str__(self):
        len_str = _data_type_reverse[self.len_dtype]
        val_str = _data_type_reverse[self.val_dtype]
        return 'property list %s %s %s' % (len_str, val_str, self.name)

    def __repr__(self):
        return 'PlyListProperty(%r, %r, %r)' % (self.name, _lookup_type(self.len_dtype), _lookup_type(self.val_dtype))

