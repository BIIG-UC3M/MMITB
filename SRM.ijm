#@String image_path
#@int q
#@String averages
#@String threeD
#@String image_out_path
print (File.isDirectory(image_path));
if (File.isDirectory(image_path) == "1") {
  run("Image Sequence...", "open="+image_path+" sort");
} else {
  open(image_path);
}
run("8-bit");
run("Statistical Region Merging", "q="+q+" "+averages+" "+threeD);
run("Save", "save="+image_out_path);
exit ("No argument!");
