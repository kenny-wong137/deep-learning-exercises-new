wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
tar xf images.tar.gz
tar xf annotations.tar.gz
mkdir predictions
python model.py images annotations/xmls predictions
