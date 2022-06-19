import os



def check_path(path):
    """
    -->
        function that checks if file exists and if so, asks the user if they want to override it.

        Parameters:
            -----------
                path: str -> path to check

    """
    if os.path.exists(path):
        protection = input(f"This file already exists. Are you sure you want to override?\nType 'Yes' to continue: ")
        if protection == 'Yes':
            return True
        else:
            print("\nCanceling Operation...\n")
            return False
    else:
        return True