#@String image_path
#@int q
#@String averages
#@String threeD
run("Image Sequence...", "open="+image_path+" sort");
run("8-bit");
run("Statistical Region Merging", "q="+q+" "+averages+" "+threeD);
saveAs("Tiff", "/tmp/outSRM.tif");
exit ("No argument!");
