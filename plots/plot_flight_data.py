import sys
import csv
import pandas

from pylab import plotfile, show, gca, rcParams
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join

for fn in listdir(sys.argv[1]):
    filename = join(sys.argv[1], fn)

    if (not isfile(filename)):
        continue

    print "reading from file '{}'".format(filename)

    data = pandas.read_csv(filename, comment='#', delimiter=", ");

#for property, value in vars(data).iteritems():
#        print property, ": ", value

    print data.IVV_R

    plt.figure(figsize=(20,15))

    plt.subplot(911)
    plt.plot(data.VIB_N11, label="VIB_N11")
    plt.plot(data.VIB_N12, label="VIB_N12")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.subplots_adjust(right=0.75)

    plt.subplot(912)
    plt.plot(data.VIB_N21, label="VIB_N21")
    plt.plot(data.VIB_N22, label="VIB_N22")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.subplots_adjust(right=0.75)

    plt.subplot(913)
    plt.plot(data.IVV_R, label="IVV_R")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.subplots_adjust(right=0.75)

    plt.subplot(914)
    plt.plot(data.BLD_PRS1, label="BLD_PRS1")
    plt.plot(data.BLD_PRS2, label="BLD_PRS2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.subplots_adjust(right=0.75)

    plt.subplot(915)
    plt.plot(data.OIL_PRS_L, label="OIL_PRS_L")
    plt.plot(data.OIL_PRS_R, label="OIL_PRS_R")
    plt.plot(data.OIL_TMP1, label="OIL_TMP1")
    plt.plot(data.OIL_TMP2, label="OIL_TMP2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.subplots_adjust(right=0.75)

    plt.subplot(916)
    plt.plot(data.SAT, label="SAT")
    plt.plot(data.TAT, label="TAT")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.subplots_adjust(right=0.75)

#PITCH PITCH2 ROLL ROLL_TRIP_P RUDD RUDD_TRIM_P
    plt.subplot(917)
    plt.plot(data.PITCH, label="PITCH")
    plt.plot(data.PITCH2, label="PITCH2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.subplots_adjust(right=0.75)

    plt.subplot(918)
    plt.plot(data.ROLL, label="ROLL")
    plt.plot(data.ROLL_TRIP_P, label="ROLL_TRIP_P")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.subplots_adjust(right=0.75)

    plt.subplot(919);
    plt.plot(data.RUDD, label="RUDD")
    plt.plot(data.RUDD_TRIM_P, label="RUDD_TRIM_P")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.subplots_adjust(right=0.75)
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1.0)

    #left  = 0.125  # the left side of the subplots of the figure
    #right = 0.9    # the right side of the subplots of the figure
    #bottom = 0.1   # the bottom of the subplots of the figure
    #top = 0.9      # the top of the subplots of the figure
    #wspace = 0.2   # the amount of width reserved for blank space between subplots
    #plt.hspace = 2.0   # the amount of height reserved for white space between subplots


    #plt.tight_layout()
    #plt.show()


    outname = sys.argv[2] + fn[:-5] + ".png"
    #outname = cbook.get_sample_data(filename, asfileobj=False)
    print "writing to '{}'".format(outname)
    plt.savefig(outname)
    plt.clf()

