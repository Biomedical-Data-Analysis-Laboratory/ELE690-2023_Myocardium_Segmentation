import sys
import ipdb
import get_images_and_masks as gim
#from IPython.testing.globalipapp import get_ipython

fr = "Ipython"
try:
    # Works with ipython
    #ip = get_ipython()
    if sys.version_info >= (3,10):
        get_ipython().run_line_magic("pdb", "on") # for post mortem debugging
        get_ipython().run_line_magic("load_ext", "autoreload")
        get_ipython().run_line_magic("autoreload", "2")
    else:
        get_ipython().run_line_magic("pdb on") # for post mortem debugging
        get_ipython().run_line_magic("load_ext autoreload")
        get_ipython().run_line_magic("autoreload 2")
except:
    # If cpython interpreter is used
    fr = "cpython"
print("Running "+fr+" frontend")


def main(argv):
    gim.main(argv)

 
if __name__ == "__main__":

    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        main()  # defaults as defined in ds.arg_read
