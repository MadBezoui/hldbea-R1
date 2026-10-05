import inspect

def get_caller_function():
    # Get the current frame
    current_frame = inspect.currentframe()
    # Go one level up to get the caller's frame
    caller_frame = current_frame.f_back
    # Go one more level up to get the caller of the caller's frame
    caller_of_caller_frame = caller_frame.f_back

    if caller_of_caller_frame:
        # Get the caller of the caller's code object
        caller_of_caller_code = caller_of_caller_frame.f_code
        # Get the caller of the caller's function name
        caller_of_caller_function_name = caller_of_caller_code.co_name
        
        return caller_of_caller_function_name
    else:
        return ''
