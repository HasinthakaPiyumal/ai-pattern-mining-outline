# Cluster 17

def execute_code_in_process(code: str):
    import z3
    import sympy
    import math
    import itertools
    from fractions import Fraction
    safe_globals = prepare_safe_globals()
    z3_whitelist = set(dir(z3))
    safe_globals.update({name: getattr(z3, name) for name in z3_whitelist})
    sympy_whitelist = set(dir(sympy))
    safe_globals.update({name: getattr(sympy, name) for name in sympy_whitelist})
    safe_globals.update({'z3': z3, 'sympy': sympy, 'Solver': z3.Solver, 'solver': z3.Solver, 'Optimize': z3.Optimize, 'sat': z3.sat, 'unsat': z3.unsat, 'unknown': z3.unknown, 'Real': z3.Real, 'Int': z3.Int, 'Bool': z3.Bool, 'And': z3.And, 'Or': z3.Or, 'Not': z3.Not, 'Implies': z3.Implies, 'If': z3.If, 'Sum': z3.Sum, 'ForAll': z3.ForAll, 'Exists': z3.Exists, 'model': z3.Model, 'Symbol': sympy.Symbol, 'solve': sympy.solve, 'simplify': sympy.simplify, 'expand': sympy.expand, 'factor': sympy.factor, 'diff': sympy.diff, 'integrate': sympy.integrate, 'limit': sympy.limit, 'series': sympy.series})

    def as_numerical(x):
        if z3.is_expr(x):
            if z3.is_int_value(x) or z3.is_rational_value(x):
                return float(x.as_decimal(20))
            elif z3.is_algebraic_value(x):
                return x.approx(20)
        return float(x)
    safe_globals['as_numerical'] = as_numerical

    def Mod(x, y):
        return x % y
    safe_globals['Mod'] = Mod

    def Rational(numerator, denominator=1):
        return z3.Real(str(Fraction(numerator, denominator)))
    safe_globals['Rational'] = Rational
    output_buffer = io.StringIO()
    with contextlib.redirect_stdout(output_buffer):
        try:
            exec(code, safe_globals, {})
        except Exception:
            return ('error', traceback.format_exc())
    return ('success', output_buffer.getvalue())

def prepare_safe_globals():
    safe_globals = {'print': print, '__builtins__': {'True': True, 'False': False, 'None': None, 'abs': abs, 'float': float, 'int': int, 'len': len, 'max': max, 'min': min, 'round': round, 'sum': sum, 'complex': complex}}
    safe_globals.update({'log': math.log, 'log2': math.log2, 'sqrt': math.sqrt, 'exp': math.exp, 'sin': math.sin, 'cos': math.cos, 'tan': math.tan, 'pi': math.pi, 'e': math.e})
    safe_globals['I'] = complex(0, 1)
    safe_globals['Complex'] = complex
    return safe_globals

