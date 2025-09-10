# Download Sintel
mkdir -p /data/wanghaoxuan/sintel
cd /data/wanghaoxuan/sintel
# images
wget --no-proxy http://files.is.tue.mpg.de/sintel/MPI-Sintel-training_images.zip
# depth & cameras
wget --no-proxy http://files.is.tue.mpg.de/jwulff/sintel/MPI-Sintel-depth-training-20150305.zip
# flow
wget --no-proxy http://files.is.tue.mpg.de/sintel/MPI-Sintel-training_extras.zip
# unzip all
find . -name "*.zip" -exec unzip -o -q {} \;
# remove all zip files
find . -name "*.zip" -exec rm {} \;
cd /home/wanghaoxuan/SVD-pi3/Pi3-evaluation
echo "Download Sintel done."