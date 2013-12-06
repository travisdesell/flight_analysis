flight_analyis
===============

Methods for retreiving, classifying and analyzing UND flight data.



Installation instructions:

git clone https://github.com/travisdesell/flight_analyis.git

cd flight_analysis

-- Here, you might need to edit the line in the .git/config file:
--    url =  https://github.com/travisdesell/flight_analyis.git
-- to
--     ssh://git@github.com/travisdesell/flight_analyis.git
-- to be able to push and correctly pull the submodules.
--
-- You may need to do this in the ./tao/.git/config and
-- ./tao/undvc_common/.git/config files as well.

git submodule init
git submodule update

cd tao

git checkout master
git submodule init
git submodule update

cd undvc_common
git checkout master

This should check out all the required code from git.  Then to
compile:

cd ..
cd ..
mkdir build
cd build
cmake ..
make
