#!/bin/bash
mencoder mf://./$1*.png  -fps 5 -ofps 5 -o video.flv -of lavf -ovc lavc -lavcopts vcodec=flv:keyint=50:vbitrate=700:mbd=2:mv0:trell:v4mv:cbp:last_pred=3 -vf yadif -sws 9
