import os



def check_path(path, OVERRIDE=False):
    """
    -->
        function that checks if file exists and if so, asks the user if they want to override it.

        Parameters:
            -----------
                path: str -> path to check

    """
    if os.path.exists(path):
        if OVERRIDE:
            print(f"Overriding the following file: {path}")
            return True
        else:
            decision = input(f"This following path already exists:\n '{path}'\nAre you sure you want to override?\n Continue? [y/n]: ")
            
            if decision == 'y':
                return True
                print("The process has been continued.")
            elif decision == 'n':
                print("The process has been halted.")
                return False
            else:
                print("You did not enter a valid option. \nCanceling Operation...")
                return False
    else:
        return True


# def overwrite_protection(path):

#         if os.path.exists(outputfp):
#             print(f"File {outputfp} already exists.")
#             print("Are you sure you want to continue and overwrite the file?")
#             decision = input('Continue? [y/n]')

#             if decision == 'y':
#                 df.to_csv(outputfp, index = False)
#                 print(f"file has been written to: {outputfp}")
#             elif decision == 'n':
#                 print("The process has been halted.")
#             else:
#                 print("You did not enter a valid option.\nThe process has halted.")
#         else:
#             df.to_csv(outputfp, index = False)
#             print(f"file has been written to: {outputfp}")