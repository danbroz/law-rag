#!/bin/bash
wget https://uscode.house.gov/download/releasepoints/us/pl/119/1/xml_uscAll@119-1.zip
mkdir data
unzip xml_uscAll@119-1.zip -d data
cd data
cat usc05.xml usc05A.xml >usc05.xml && rm usc05A.xml
cat usc11.xml usc11a.xml >usc11.xml && rm usc11a.xml
cat usc18.xml usc18a.xml >usc18.xml && rm usc18a.xml
cat usc28.xml usc28a.xml >usc28.xml && rm usc28a.xml
cat usc50.xml usc50A.xml >usc50.xml && rm usc50A.xml
