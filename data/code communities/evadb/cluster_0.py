# Cluster 0

def convert_file(name, source, dest):
    s = open(source, 'rb').read().decode('utf-8')
    m = index.search(s)
    if m:
        s = index_sub(m) + '.. _design-{0}:\n\n'.format(name) + s
    s = mode.sub('', s)
    s = prefix.sub('.. mps:prefix:: \\1', s)
    s = rst_tag.sub('', s)
    s = mps_tag.sub(':mps:tag:`\\1`', s)
    s = mps_ref.sub(':mps:ref:`\\1`', s)
    s = typedef.sub('.. c:type:: \\1', s)
    s = funcdef.sub('.. c:function:: \\1', s)
    s = macrodef.sub('.. c:macro:: \\1', s)
    s = typename.sub(':c:type:`\\1`', s)
    s = func.sub(':c:func:`\\1`', s)
    s = macro.sub(':c:macro:`\\1`', s)
    s = secnum.sub(secnum_sub, s)
    s = citation.sub(citation_sub, s)
    s = design_ref.sub('\\1.html', s)
    s = design_frag_ref.sub('\\1.html#design.mps.\\2.\\3', s)
    s = history.sub('', s)
    s = '.. highlight:: none\n\n' + s
    try:
        os.makedirs(os.path.dirname(dest))
    except:
        pass
    with open(dest, 'wb') as out:
        out.write(s.encode('utf-8'))

def index_sub(m):
    s = '\n.. index::\n'
    for term in index_term.finditer(m.group(1)):
        s += '   %s: %s\n' % (term.group(1), term.group(2))
    s += '\n'
    return s

def convert_updated(app):
    for design in glob.iglob('../design/*.txt'):
        name = os.path.splitext(os.path.basename(design))[0]
        if name == 'index':
            continue
        converted = 'source/design/%s.rst' % name
        if newer(design, converted):
            app.info('converting design %s' % name)
            convert_file(name, design, converted)
    for diagram in glob.iglob('../design/*.svg'):
        target = os.path.join('source/design/', os.path.basename(diagram))
        if newer(diagram, target):
            shutil.copyfile(diagram, target)

def newer(src, target):
    """Return True if src is newer (that is, modified more recently) than
    target, False otherwise.

    """
    return not os.path.isfile(target) or os.path.getmtime(target) < os.path.getmtime(src) or os.path.getmtime(target) < os.path.getmtime(__file__)

