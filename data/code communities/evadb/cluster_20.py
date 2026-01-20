# Cluster 20

def _mk_label(mapper, show_operations, show_attributes, show_datatypes, show_inherited, bordersize):
    html = '<<TABLE CELLSPACING="0" CELLPADDING="1" BORDER="0" CELLBORDER="%d" ALIGN="LEFT"><TR><TD><FONT POINT-SIZE="10">%s</FONT></TD></TR>' % (bordersize, mapper.class_.__name__)

    def format_col(col):
        colstr = '+%s' % col.name
        if show_datatypes:
            colstr += ' : %s' % col.type.__class__.__name__
        return colstr
    if show_attributes:
        if not show_inherited:
            cols = [c for c in mapper.columns if c.table == mapper.tables[0]]
        else:
            cols = mapper.columns
        html += '<TR><TD ALIGN="LEFT">%s</TD></TR>' % '<BR ALIGN="LEFT"/>'.join((format_col(col) for col in cols))
    else:
        [format_col(col) for col in sorted(mapper.columns, key=lambda col: not col.primary_key)]
    if show_operations:
        html += '<TR><TD ALIGN="LEFT">%s</TD></TR>' % '<BR ALIGN="LEFT"/>'.join(('%s(%s)' % (name, ', '.join((default is _mk_label and '%s' % arg or '%s=%s' % (arg, repr(default)) for default, arg in zip((func.func_defaults and len(func.func_code.co_varnames) - 1 - (len(func.func_defaults) or 0) or func.func_code.co_argcount - 1) * [_mk_label] + list(func.func_defaults or []), func.func_code.co_varnames[1:])))) for name, func in mapper.class_.__dict__.items() if isinstance(func, types.FunctionType) and func.__module__ == mapper.class_.__module__))
    html += '</TABLE>>'
    return html

def format_col(col):
    colstr = '+%s' % col.name
    if show_datatypes:
        colstr += ' : %s' % col.type.__class__.__name__
    return colstr

def create_uml_graph(mappers, show_operations=True, show_attributes=True, show_inherited=True, show_multiplicity_one=False, show_datatypes=True, linewidth=1.0, font='Bitstream-Vera Sans'):
    graph = pydot.Dot(prog='neato', mode='major', overlap='0', sep='0.01', dim='3', pack='True', ratio='.75')
    relations = set()
    for mapper in mappers:
        graph.add_node(pydot.Node(escape(mapper.class_.__name__), shape='plaintext', label=_mk_label(mapper, show_operations, show_attributes, show_datatypes, show_inherited, linewidth), fontname=font, fontsize='8.0'))
        if mapper.inherits:
            graph.add_edge(pydot.Edge(escape(mapper.inherits.class_.__name__), escape(mapper.class_.__name__), arrowhead='none', arrowtail='empty', style='setlinewidth(%s)' % linewidth, arrowsize=str(linewidth)))
        for loader in mapper.iterate_properties:
            if isinstance(loader, RelationshipProperty) and loader.mapper in mappers:
                if hasattr(loader, 'reverse_property'):
                    relations.add(frozenset([loader, loader.reverse_property]))
                else:
                    relations.add(frozenset([loader]))
    for relation in relations:
        args = {}

        def multiplicity_indicator(prop):
            if prop.uselist:
                return ' *'
            if hasattr(prop, 'local_side'):
                cols = prop.local_side
            else:
                cols = prop.local_columns
            if any((col.nullable for col in cols)):
                return ' 0..1'
            if show_multiplicity_one:
                return ' 1'
            return ''
        if len(relation) == 2:
            src, dest = relation
            from_name = escape(src.parent.class_.__name__)
            to_name = escape(dest.parent.class_.__name__)

            def calc_label(src, dest):
                return '+' + src.key + multiplicity_indicator(src)
            args['headlabel'] = calc_label(src, dest)
            args['taillabel'] = calc_label(dest, src)
            args['arrowtail'] = 'none'
            args['arrowhead'] = 'none'
            args['constraint'] = False
        else:
            prop, = relation
            from_name = escape(prop.parent.class_.__name__)
            to_name = escape(prop.mapper.class_.__name__)
            args['headlabel'] = '+%s%s' % (prop.key, multiplicity_indicator(prop))
            args['arrowtail'] = 'none'
            args['arrowhead'] = 'vee'
        graph.add_edge(pydot.Edge(from_name, to_name, fontname=font, fontsize='7.0', style='setlinewidth(%s)' % linewidth, arrowsize=str(linewidth), **args))
    return graph

def escape(name):
    return '"%s"' % name

def calc_label(src, dest):
    return '+' + src.key + multiplicity_indicator(src)

def multiplicity_indicator(prop):
    if prop.uselist:
        return ' *'
    if hasattr(prop, 'local_side'):
        cols = prop.local_side
    else:
        cols = prop.local_columns
    if any((col.nullable for col in cols)):
        return ' 0..1'
    if show_multiplicity_one:
        return ' 1'
    return ''

def show_uml_graph(*args, **kwargs):
    from cStringIO import StringIO
    from PIL import Image
    iostream = StringIO(create_uml_graph(*args, **kwargs).create_png())
    Image.open(iostream).show(command=kwargs.get('command', 'gwenview'))

