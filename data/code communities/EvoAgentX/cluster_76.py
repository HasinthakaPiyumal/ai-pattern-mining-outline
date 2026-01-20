# Cluster 76

def run_python_interpreter_examples():
    """Run all examples using the Python InterpreterToolkit"""
    print('\n===== PYTHON INTERPRETER EXAMPLES =====\n')
    interpreter_toolkit = PythonInterpreterToolkit(project_path=os.getcwd(), directory_names=['examples', 'evoagentx'], allowed_imports={'os', 'sys', 'time', 'datetime', 'math', 'random', 'platform', 'matplotlib.pyplot', 'numpy'})
    interpreter = interpreter_toolkit.python_interpreter
    run_simple_hello_world(interpreter)
    run_math_example(interpreter)
    run_platform_info(interpreter)
    run_script_execution(interpreter)
    run_dynamic_code_generation(interpreter)
    run_visualization_example(interpreter)

def run_simple_hello_world(interpreter):
    """
    Run a simple Hello World example using the provided interpreter.
    
    Args:
        interpreter: An instance of a code interpreter
    """
    code = '\nprint("Hello, World!")\nprint("This code is running inside a secure Python interpreter.")\n'
    result = interpreter.execute(code, 'python')
    print('\nSimple Hello World Result:')
    print('-' * 50)
    print(result)
    print('-' * 50)

def run_math_example(interpreter):
    """
    Run a math example using the provided interpreter.
    
    Args:
        interpreter: An instance of a code interpreter
    """
    code = '\nprint("Running math operations...")\n\n# Using math library\nimport math\nprint(f"The value of pi is: {math.pi:.4f}")\nprint(f"The square root of 16 is: {math.sqrt(16)}")\nprint(f"The value of e is: {math.e:.4f}")\n'
    result = interpreter.execute(code, 'python')
    print('\nMath Example Result:')
    print('-' * 50)
    print(result)
    print('-' * 50)

def run_platform_info(interpreter):
    """
    Run a platform info example using the provided interpreter.
    
    Args:
        interpreter: An instance of a code interpreter
    """
    code = '\nprint("Getting platform information...")\n\n# System information\nimport platform\nimport sys\n\nprint(f"Python version: {platform.python_version()}")\nprint(f"Platform: {platform.system()} {platform.release()}")\nprint(f"Processor: {platform.processor()}")\nprint(f"Implementation: {platform.python_implementation()}")\n'
    result = interpreter.execute(code, 'python')
    print('\nPlatform Info Result:')
    print('-' * 50)
    print(result)
    print('-' * 50)

def run_script_execution(interpreter):
    """
    Run a script file using the execute_script method of the interpreter.
    
    Args:
        interpreter: An instance of a code interpreter
    """
    script_path = os.path.join(os.getcwd(), 'examples', 'tools', 'hello_world.py')
    if not os.path.exists(script_path):
        print(f'Error: Script file not found at {script_path}')
        return
    print(f'Executing script file: {script_path}')
    result = interpreter.execute_script(script_path, 'python')
    print('\nScript Execution Result:')
    print('-' * 50)
    print(result)
    print('-' * 50)

def run_dynamic_code_generation(interpreter):
    """
    Run an example that demonstrates dynamic code generation and execution.
    
    Args:
        interpreter: An instance of a code interpreter
    """
    code = '\nprint("Generating and executing code dynamically...")\n\n# Generate a function definition\nfunction_code = \'\'\'\ndef calculate_factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * calculate_factorial(n-1)\n\'\'\'\n\n# Execute the generated code to define the function\nexec(function_code)\n\n# Now use the dynamically defined function\nfor i in range(1, 6):\n    print(f"Factorial of {i} is {calculate_factorial(i)}")\n'
    result = interpreter.execute(code, 'python')
    print('\nDynamic Code Generation Result:')
    print('-' * 50)
    print(result)
    print('-' * 50)

def run_visualization_example(interpreter):
    """
    Run an example that would generate a visualization if matplotlib was allowed.
    This demonstrates handling imports that might not be allowed.
    
    Args:
        interpreter: An instance of a code interpreter
    """
    code = '\nprint("Attempting to create a simple visualization...")\n\ntry:\n    import matplotlib.pyplot as plt\n    import numpy as np\n    \n    # Generate some data\n    x = np.linspace(0, 10, 100)\n    y = np.sin(x)\n    \n    # Create a plot\n    plt.figure(figsize=(8, 4))\n    plt.plot(x, y)\n    plt.title("Sine Wave")\n    plt.xlabel("x")\n    plt.ylabel("sin(x)")\n    plt.grid(True)\n    \n    # Save the plot (would work if matplotlib was available)\n    plt.savefig("examples/output/sine_wave.png")\n    plt.close()\n    \n    print("Visualization created and saved as \'examples/output/sine_wave.png\'")\nexcept ImportError as e:\n    print(f"Import error: {e}")\n    print("Note: This example requires matplotlib to be in the allowed_imports.")\n'
    result = interpreter.execute(code, 'python')
    print('\nVisualization Example Result:')
    print('-' * 50)
    print(result)
    print('-' * 50)

def run_docker_interpreter_examples():
    """Run all examples using the Docker InterpreterToolkit"""
    print('\n===== DOCKER INTERPRETER EXAMPLES =====\n')
    print('Running Docker interpreter examples...')
    try:
        interpreter_toolkit = DockerInterpreterToolkit(image_tag='python:3.9-slim', print_stdout=True, print_stderr=True, container_directory='/app')
        interpreter = interpreter_toolkit.docker_interpreter
        run_simple_hello_world(interpreter)
        run_math_example(interpreter)
        run_platform_info(interpreter)
        run_script_execution(interpreter)
        run_dynamic_code_generation(interpreter)
    except Exception as e:
        print(f'Error running Docker interpreter examples: {str(e)}')
        print('Make sure Docker is installed and running on your system.')
        print('You may need to pull the python:3.9-slim image first using: docker pull python:3.9-slim')

def main():
    """Main function to run interpreter examples"""
    print('===== CODE INTERPRETER EXAMPLES =====')
    run_python_interpreter_examples()
    run_docker_interpreter_examples()
    print('\n===== ALL INTERPRETER EXAMPLES COMPLETED =====')

