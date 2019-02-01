# In this demo we will look at some particular capabilities of FBL.
# Let us suppose that we have found information on a specific neuron:
# http://flycircuit.tw/modules.php?name=clearpage&op=detail_table&neuron=Gad1-F-400259

# Say we find a similar neuron; we can add them to our workspace with:
nm[0].addByUname(['Cha-F-800086','Gad1-F-400259'])

# We can save these neurons for later:
nm[0].createTag('PLACEHOLDER')
# We can then load them back up:
nm[0].loadTag('PLACEHOLDER')
# This allows us to store the state of a particular circuit for later.

# If we are using the GFX interface, we can click on neurons in the 3D view to see their presynaptic and postsynaptic targets. Otherwise, we can use the GetInfo function:
nm[0].getInfo(nm[0].uname_to_rid['Cha-F-800086'])